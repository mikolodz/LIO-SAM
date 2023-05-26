#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>


class ImuEstimation : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    ImuProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        imuEstiamtion();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void imuEstiamtion()
    {
        int N_length = 5000;
        float delta_t = 0.01f;
        float wie = 7.2921158f*0.0001f;
        float* g_ned;
        g_ned = zeros(3,1);
        g_ned[2] = -9.80665f;
        int a = 6378136;
        float e2 = 6.99437999014f*0.001f;
        float* gyro_imu = zeros(3,1);
        gyro_imu[0] = RealDataIn->gyro_out[0];
        gyro_imu[1] = RealDataIn->gyro_out[1];
        gyro_imu[2] = RealDataIn->gyro_out[2];
        float* omega_imu = mat_skew(gyro_imu);
        dcm_matrix = zeros(3,3);
        float* delta_matrix = ones(3,delta_t);
        int counter = 0;
        float* tmp_matrix;
        matmul("NN",3,3,3,1.0f,pre_dcm,omega_imu,0,tmp_matrix);
        matmul("NN",3,3,3,1.0f,delta_matrix,tmp_matrix,0,tmp_matrix);
        for(counter = 0; counter <= 8;counter++)
        {
            dcm_matrix[counter] = pre_dcm[counter] + tmp_matrix[counter];
        }
        free(omega_imu);
        free(delta_matrix);
        free(tmp_matrix);
        float* dcm_1 = zeros(1,3);
        float* dcm_2 = zeros(1,3);
        float* dcm_3 = zeros(1,3);
         
        // Init kalman for DCM Estimate
        float* x_dcm3_k = zeros(3,1);
        float* x_dcm3_k_ = zeros(3,1);
        float* P_dcm3_k = zeros(3,3);
        float* P_dcm3_k_ = zeros(3,3);

        float* x_dcm1_k = zeros(3,1);
        float* x_dcm1_k_ = zeros(3,1);
        float* P_dcm1_k = zeros(3,3);
        float* P_dcm1_k_ = zeros(3,3);

        float q_omegax = 2.0369e-11;
        float q_omegay = 0.8e-11;
        float q_omegaz = 6.4e-11;

        float sigma_alfax = 4.4436e-07;
        float sigma_alfay = 3.0942e-07;
        float sigma_alfaz = 2.8166e-07;

        float sigma_mx = 6.2829e-06;
        float sigma_my = 2.1239e-06;
        float sigma_mz = 2.0318e-06;

        for(counter = 0; counter < 3;counter++)
        {
            dcm_1[counter] = dcm_matrix[counter];
            dcm_2[counter] = dcm_matrix[counter+3];
            dcm_3[counter] = dcm_matrix[counter+6];
        }	
        for(counter = 0; counter < 3;counter++)
        {
            dcm_1[counter] = dcm_1[counter]/norm(dcm_1,3);
            dcm_matrix[counter] = dcm_1[counter];
            dcm_2[counter] = dcm_2[counter]/norm(dcm_2,3);
            dcm_matrix[counter+3] = dcm_2[counter];
            dcm_3[counter] = dcm_3[counter]/norm(dcm_3,3);
            dcm_matrix[counter+6] = dcm_3[counter];
        }	
        free(dcm_1);
        free(dcm_2);
        free(dcm_3);

        double* Pre_DCM       = My_NewDCM(-0.51964474*pi/180,-7.4047756*pi/180,-165.97565*pi/180);
        double* DCM                   = My_NewDCM(-0.51964474*pi/180,-7.4047756*pi/180,-165.97565*pi/180);
        // sigma
        double* sigma_accl            = [1 1 1];
        double* sigma_gyro            = [0.00000001 0.000000001 0.00000001];
        double* sigma_r1              = [1e-13 1e-13 1e-10];
        double* sigma_r2              = [1e-11 1e-11 1e-10];
        int k                     = 1;
        double* Q  = diag([sigma_accl sigma_gyro]);
        x_dcm1_k = [DCM(1,1); DCM(1,2); DCM(1,3)];
        x_dcm3_k = [DCM(3,1); DCM(3,2); DCM(3,3)];

        AngleData->angle.roll = atan2(dcm_matrix[7],dcm_matrix[8]);
        AngleData->angle.pitch = -atan(dcm_matrix[6]/(sqrt(1-(dcm_matrix[6]*dcm_matrix[6]))));
        AngleData->angle.yaw = atan2(dcm_matrix[3],dcm_matrix[0]);
        matcpy(pre_dcm,dcm_matrix,3,3);
            

        // INS dynamic model
        for (int i = 1, i <= N_length, i++)
        {
            // Init
            float M = a/(sqrt(1 - e2*sin(My_INS.lat(i))^2));
            float N = a*(1-e2)/(sqrt((1 - e2*sin(My_INS.lat(i))^2)^3));
            
            // gyro
            double* gyro_IMU   = [Gyro.x(i); Gyro.y(i); Gyro.z(i)];
            double* gyro_Earth = [wie*cos(My_INS.lat(i));...
                                0;...
                                -wie*sin(My_INS.lat(i))];
            double* gyro_NED   = [My_INS.vE(i)/(N+My_INS.h(i));...
                                -My_INS.vN(i)/(M+My_INS.h(i));...
                                -My_INS.vE(i)*tan(My_INS.lat(i))/(N+My_INS.h(i))];              
                    
            // Skew symmetric matrix
            float OMEGA_IMU   = My_SkewSymmetric(gyro_IMU);
            float OMEGA_Earth = My_SkewSymmetric(gyro_Earth);
            float OMEGA_NED   = My_SkewSymmetric(gyro_NED);    
            // angular estimate update
            float T = 0.01;
            // Row 3 of DCM Maxtrix
            float* A_dcm3 = eye(3) - OMEGA_IMU*delta_t;
            float* H_dcm3 = eye(3);
            float* W_dcm3 = [  0           -DCM(3,3)*T     DCM(3,2)*T;...
                            DCM(3,3)*T  0               -DCM(3,1)*T;...
                            -DCM(3,2)*T DCM(3,1)*T      0];
            float* V_dcm3 = eye(3);
            float* Q_dcm3 = diag([q_omegax, q_omegay, q_omegaz]);
            float* R_dcm3 = diag([sigma_alfax, sigma_alfay, sigma_alfaz]);
            float* z_dcm3 = [Accl.x(i); Accl.y(i); Accl.z(i)];
            // predict stage
            x_dcm3_k_ = A_dcm3*x_dcm3_k;
            P_dcm3_k_ = A_dcm3*P_dcm3_k*A_dcm3' + W_dcm3*Q_dcm3*W_dcm3';
            // update stage
            double* K_dcm3 = P_dcm3_k_*H_dcm3'/(H_dcm3*P_dcm3_k_*H_dcm3' + V_dcm3*R_dcm3*V_dcm3'); % kalman gain
            double* x_dcm3_k = x_dcm3_k_ + K_dcm3*(z_dcm3 - H_dcm3*x_dcm3_k_);
            P_dcm3_k = (eye(3) - K_dcm3*H_dcm3)*P_dcm3_k_;
                
                
            // Row 1 of DCM Maxtrix
            double* A_dcm1 = eye(3) - OMEGA_IMU*delta_t;
            float Ic = 0.987642;
            float Is = 0.156722;
            double* H_dcm1 = Ic*eye(3);
            double* W_dcm1 = [  0           -DCM(1,3)*T     DCM(1,2)*T;...
                        DCM(1,3)*T  0               -DCM(1,1)*T;...
                        -DCM(1,2)*T DCM(1,1)*T      0];
            double* V_dcm1 = eye(3);
            double* Q_dcm1 = diag([q_omegax, q_omegay, q_omegaz]);
            double* R_dcm1 = diag([sigma_mx, sigma_my, sigma_mz]);
            double* z_dcm1 = [Mag.x(i); Mag.y(i); Mag.z(i)] - x_dcm3_k*Is;
            // predict stage
            double* x_dcm1_k_ = A_dcm1*x_dcm1_k;
            P_dcm1_k_ = A_dcm1*P_dcm1_k*A_dcm1' + W_dcm1*Q_dcm1*W_dcm1';
            // update stage
            double* K_dcm1 = P_dcm1_k_*H_dcm1'/(H_dcm1*P_dcm1_k_*H_dcm1' + V_dcm1*R_dcm1*V_dcm1'); % kalman gain
            double* x_dcm1_k = x_dcm1_k_ + K_dcm1*(z_dcm1 - H_dcm1*x_dcm1_k_);
            P_dcm1_k = (eye(3) - K_dcm1*H_dcm1)*P_dcm1_k_;
                
            double* x_dcm2 = cross(x_dcm3_k', x_dcm1_k');
            DCM = [x_dcm1_k'; x_dcm2; x_dcm3_k'];

            My_INS.pitch(i) = -atan(DCM(3,1)/sqrt(1-DCM(3,1)^2))*180/pi;
                
            if(i >= 1400) 
            {
                My_INS.pitch(i) = My_INS.pitch(i) + 3.2;
            }

            My_INS.roll(i)  = atan2(DCM(3,2),DCM(3,3))*180/pi;
            My_INS.yaw(i)   = atan2(DCM(2,1),DCM(1,1))*180/pi;
            Pre_DCM = DCM;
                
            // accl update
            double* f_body = [Accl.x(i); Accl.y(i); Accl.z(i)];
            float f_ned  = (DCM + Pre_DCM)*f_body/2;
            Pre_DCM = DCM;
                
                
            // speed update
            float* v              = Pre_v + (Pre_DCM*f_body + g_ned - (2*OMEGA_Earth + OMEGA_NED)*Pre_v)*delta_t;
            My_INS.vN(i+1) = v(1);
            My_INS.vE(i+1) = v(2);
            My_INS.vD(i+1) = v(3);


            // Position update
            My_INS.h(i+1)     = My_INS.h(i) - delta_t/2*(My_INS.vD(i+1) + My_INS.vD(i));
            float Pre_M             = M;
            M                 = a/(sqrt(1 - e2*sin(My_INS.lon(i+1))^2));
            float Pre_N             = N;
            N                 = a*(1-e2)/(sqrt(1 - e2*sin(My_INS.lon(i+1))^2))^3;
            My_INS.lon(i+1)   = My_INS.lon(i) + delta_t/2*(My_INS.vN(i)/(Pre_M + My_INS.h(i)) + My_INS.vN(i+1)/(M + My_INS.h(i+1)));
            My_INS.lat(i+1) = My_INS.lat(i) + delta_t/2*(My_INS.vE(i)/((Pre_N+My_INS.h(i))*cos(My_INS.lon(i)))+My_INS.vE(i+1)/((N + My_INS.h(i))*cos(My_INS.lon(i+1))));
            r(1) =  My_INS.lat(i+1);
            r(2) =  My_INS.lon(i+1);
            r(3) =  My_INS.h(i+1);
        }
            
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Imu Estimation Start.\033[0m");
    imuEstimation();
    
    return 0;
}
