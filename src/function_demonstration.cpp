#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
using namespace std;
using namespace cv;
typedef pcl::PointXYZ PCType;

class function_demonstration
{
public:
  ros::NodeHandle nh_;

  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime< sensor_msgs::Image, sensor_msgs::Image> >* sync_input_2_; 
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub;

  ros::Subscriber camera_info_sub;
  boost::array<double,9> camera_info;
  ros::Publisher fingertip_3d_pub;
  ros::Publisher pointcloud_pub;  
  ros::Subscriber fingertip_sub;
  ros::Subscriber control_signal_sub;

  //bool cam_info_ready;
  float fx;
  float fy;
  float cx;
  float cy;
  float invfx;
  float invfy;
  tf::TransformListener listener;
  string tf_reference_frame;

  geometry_msgs::PointStamped current_fingertip;
  pcl::PointCloud<PCType>::Ptr current_cloud;
  pcl::PointCloud<PCType>::Ptr object_cloud;

  bool got_object_flag;
  bool got_function_part_flag;
  double timer_start;
  std::vector<geometry_msgs::Point> traj;

  function_demonstration()//:camera_info{0}
  {
    rgb_sub.subscribe(nh_, "/hsrb/head_rgbd_sensor/rgb/image_rect_color", 1);
    depth_sub.subscribe(nh_, "/hsrb/head_rgbd_sensor/depth_registered/image", 1);
    sync_input_2_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime< sensor_msgs::Image,sensor_msgs::Image> >(1);
    sync_input_2_->connectInput(rgb_sub,depth_sub);
    sync_input_2_->registerCallback(boost::bind(&function_demonstration::callback, this, _1, _2));
    
    camera_info_sub = nh_.subscribe("/hsrb/head_rgbd_sensor/rgb/camera_info",1, &function_demonstration::camera_info_cb,this);

    fingertip_sub = nh_.subscribe("/mechknownet/fingertip", 1, &function_demonstration::fingertip_callback,this);

    control_signal_sub = nh_.subscribe("/mechknownet/control_signal", 1, &function_demonstration::signal_callback,this);

    fingertip_3d_pub = nh_.advertise<geometry_msgs::PointStamped> ("/mechknownet/fingertip_3d", 1);
    pointcloud_pub = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/mechknownet/object", 1);
	//cam_info_ready = false;
	tf_reference_frame = "/base_link";
	got_object_flag = false;
	got_function_part_flag = false;
	timer_start = -1;
  }

  
  void camera_info_cb(const sensor_msgs::CameraInfoPtr& camInfo)
  {
    camera_info = camInfo->K;
    fx = camera_info[0];
    fy = camera_info[4];
    cx = camera_info[2];
    cy = camera_info[5];
    invfx = 1.0f/fx;
    invfy = 1.0f/fy;
    camera_info_sub.shutdown();
	//cam_info_ready = true;
  }

  void publish_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
  {
	sensor_msgs::PointCloud2 pc2;
	pcl::PCLPointCloud2::Ptr pcl_pc_2(new pcl::PCLPointCloud2());
	pcl::toPCLPointCloud2 (*cloud, *pcl_pc_2);
	pcl_conversions::fromPCL( *pcl_pc_2, pc2 );
	pc2.header.frame_id = tf_reference_frame;
	pointcloud_pub.publish(pc2);	
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cut_object_function_part(std::vector<geometry_msgs::Point> traj)
  {
	cout<<"???????????"<<traj.size()<<endl;
	float minx = 100;
	float maxx = -100;
	float miny = 100;
	float maxy = -100;
	float minz = 100;
	float maxz = -100;
	  
	for(int i=0; i<traj.size(); i++)
	  {
		if(traj[i].x < minx)
		  minx = traj[i].x;
		if(traj[i].x > maxx)
		  maxx = traj[i].x;
		if(traj[i].y < miny)
		  miny = traj[i].y;
		if(traj[i].y > maxy)
		  maxy = traj[i].y;
		if(traj[i].z < minz)
		  minz = traj[i].z;
		if(traj[i].z > maxz)
		  maxz = traj[i].z;
	  }
	minx = minx-0.03;
	miny = miny-0.03;
	minz = minz-0.03;
	maxx = maxx+0.03;
	maxy = maxy+0.03;
	maxz = maxz+0.03;	
	cout<<"!!!!!!"<<minx<<"!"<<maxx<<"!"<<miny<<"!"<<maxy<<"!"<<minz<<"!"<<maxz<<endl;
	pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);

	pcl::PointCloud <PCType>::Ptr cloud_remain (new pcl::PointCloud <PCType>);
	std::vector<int> indices;
	pcl::CropBox<PCType> boxFilter(true);
	boxFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
	boxFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));
	boxFilter.setInputCloud(object_cloud);
	boxFilter.filter(*cloud_cropped);
	boxFilter.filter(indices);
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(*cloud_cropped, *cloud_cropped, nan_indices);		
	pcl::PLYWriter ply_saver; 
	//ply_saver.write("function_cloud.ply",*cloud_cropped);    
	//cout<<"!!!!!!!!"<<cloud_cropped->points.size()<<endl;


	pcl::PointCloud<pcl::PointXYZRGB>::Ptr function_cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*cloud_cropped, *function_cloud_rgb);
	for(int i=0; i<function_cloud_rgb->points.size(); i++)
	  {
		function_cloud_rgb->points[i].r = 0;
		function_cloud_rgb->points[i].g = 255;
		function_cloud_rgb->points[i].b = 0;
	  }
	
	pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
	for (int point : indices)
	  {
		inliers->indices.push_back(point);
	  }

	pcl::ExtractIndices<PCType> extract;
	extract.setInputCloud(object_cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(*cloud_remain);
	//cout<<"2222222"<<cloud_remain->points.size()<<endl;
	//ply_saver.write("remain_cloud.ply",*cloud_remain);    

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*cloud_remain, *cloud_pub);
	for(int i=0; i<cloud_pub->points.size(); i++)
	  {
		cloud_pub->points[i].r = 255;
		cloud_pub->points[i].g = 255;
		cloud_pub->points[i].b = 255;
	  }

	//add function part to cloud_pub
	*cloud_pub += *function_cloud_rgb;
	publish_pointcloud(cloud_pub);
	cout<<"point cloud pubed"<<endl;
	
	ply_saver.write("cloud_with_label.ply",*cloud_pub);    


	return cloud_cropped;
  }

  void signal_callback(const std_msgs::String signal)
  {
	string data = signal.data;
	if(data=="get_object")
	  {
		try{
		  listener.waitForTransform(tf_reference_frame, "/head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(3.0));
		}	   
		catch(tf::TransformException ex)
		  {
			ROS_ERROR("%s",ex.what());
		  }

		pcl::PointCloud<PCType>::Ptr cloud_trans(new pcl::PointCloud<PCType>());
		pcl_ros::transformPointCloud(tf_reference_frame, *current_cloud, *cloud_trans, listener);
		cout<<"!!!!!!!!!!!!!!"<<current_fingertip.point<<endl;
		float x = current_fingertip.point.x;
		float y = current_fingertip.point.y;
		float z = current_fingertip.point.z;
		//crop point cloud
		/*
		pcl::search::Search <PCType>::Ptr tree (new pcl::search::KdTree<PCType>);
		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		pcl::ConditionAnd<PCType>::Ptr range_cond (new pcl::ConditionAnd<PCType> ());
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("x", pcl::ComparisonOps::GT, x-0.15)));
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("x", pcl::ComparisonOps::LT, x+0.15)));
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("y", pcl::ComparisonOps::GT, y-0.15)));
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("y", pcl::ComparisonOps::LT, y+0.15)));
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::GT, z)));
		//range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::LT, z+0.1)));
		pcl::ConditionalRemoval<PCType> condrem;
		condrem.setCondition (range_cond);
		condrem.setInputCloud (cloud_trans);
		condrem.setKeepOrganized(true);
		condrem.filter (*cloud_cropped);
		*/
		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		float minx = x - 0.15;
		float maxx = x + 0.15;
		float miny = y - 0.15;
		float maxy = y + 0.15;
		float minz = z;
		float maxz = z + 0.2;
		pcl::CropBox<PCType> boxFilter;
		boxFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
		boxFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));
		boxFilter.setInputCloud(cloud_trans);
		boxFilter.filter(*cloud_cropped);

		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_cropped, *cloud_cropped, nan_indices);


		object_cloud = cloud_cropped;
		//pcl::PLYWriter ply_saver; 
		//ply_saver.write("frame_updated_cloud.ply",*cloud_cropped);    
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*cloud_cropped, *cloud_show);
		for(int i=0; i<cloud_show->points.size(); i++)
		  {
			cloud_show->points[i].r = 255;
			cloud_show->points[i].g = 80;
			cloud_show->points[i].b = 80;
		  }
		publish_pointcloud(cloud_show);
		cout<<"point cloud pubed"<<endl;
		got_object_flag = true;
	  }
  }

  void fingertip_callback(const geometry_msgs::PointStamped point)
  {
	tf::TransformListener listener;
	geometry_msgs::PointStamped pt;
	pt = point;
	pt.header.stamp = ros::Time(0);
	geometry_msgs::PointStamped pt_transformed;

	try{
	  listener.waitForTransform(tf_reference_frame, "/head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(3.0));
	}

	catch(tf::TransformException ex)
	  {
		ROS_ERROR("%s",ex.what());
	  }
	//pt.header.
	listener.transformPoint(tf_reference_frame, pt, pt_transformed);
	fingertip_3d_pub.publish(pt_transformed);
	current_fingertip = pt_transformed;

	if(got_object_flag && !got_function_part_flag) //record fingertip trajectory
	  {
		Eigen::Vector4f pcaCentroid;
		pcl::compute3DCentroid(*object_cloud, pcaCentroid);
		float x = pcaCentroid(0);
		float y = pcaCentroid(1);
		float z = pcaCentroid(2);
		float dis = abs(x-current_fingertip.point.x)*abs(x-current_fingertip.point.x)
		  +abs(y-current_fingertip.point.y)*abs(y-current_fingertip.point.y)
		  +abs(z-current_fingertip.point.z)*abs(z-current_fingertip.point.z);
		float dis_sqrt = sqrt(dis);
		if (dis_sqrt < 1.5)
		  {
			if(timer_start==-1)
			  timer_start = ros::Time::now().toSec();
			else
			  {
				float duration = ros::Time::now().toSec() - timer_start;
				if (duration < 15)
				  {
					traj.push_back(current_fingertip.point);
				  }
				else //got finger trajectory
				  {
					cout<<"got finger traj"<<endl;
					pcl::PointCloud<PCType>::Ptr object_function_part = cut_object_function_part(traj);
					got_function_part_flag = true;
				  }
			  }
		  }
		cout<<"x "<<x<<" y "<<y<<" z "<<z<<" dis "<<dis_sqrt<<endl;
	  }
  }
  

  void callback(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD)
  {

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImage cv_img;
    try
      {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
      }
    catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
      {
        cv_ptrD = cv_bridge::toCvShare(msgD);
      }
    catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    cv::Mat depth;
    cv_ptrD->image.copyTo(depth);
    cv::Mat depth_f;

    cv::Mat RGBImg;
    cv_ptrRGB->image.copyTo(RGBImg);
    //cv::cvtColor(RGBImg,RGBImg,CV_BGR2RGB);

    if (depth.type()==2)
      depth.convertTo(depth_f,CV_32FC1, 1.0/1000);
    else if (depth.type()==5)
      depth_f = depth;
    else
      {
        cout<<"unknown depth Mat type"<<endl;
        return;
      }

    pcl::PointCloud<PCType>::Ptr cloud(new pcl::PointCloud<PCType>());
	cloud->header.frame_id = cv_ptrD->header.frame_id;
    for(int i = 0; i < depth_f.cols; i++) {
      for(int j = 0; j < depth_f.rows; j++) {
		float z =  depth_f.at<float>(j,i);
		if(z>0)
		  {
			PCType point;
			point.x = (i-cx)*z*invfx;
			point.y = (j-cy)*z*invfy;
			point.z = z;
			cloud->points.push_back(point);
		  }    
      }
    }
	current_cloud = cloud;
  } //end void callback
};
int main(int argc, char** argv)
{
  ros::init(argc, argv, "rgb_depth");
  function_demonstration fd;
  ros::spin();

  return 0;
}
