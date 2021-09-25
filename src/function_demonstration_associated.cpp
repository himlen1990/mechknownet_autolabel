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

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

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
  ros::Publisher marker_pub;  
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
	marker_pub = nh_.advertise<visualization_msgs::MarkerArray>("mechknownet_function_text_marker", 1);
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


  void publish_text(geometry_msgs::Point location)
  {
	visualization_msgs::MarkerArray  marker_array;
	std::vector<visualization_msgs::Marker> vMarker;
	visualization_msgs::Marker marker;
	marker.header.frame_id = "base_link";
	marker.header.stamp = ros::Time::now();
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::Marker::ADD;
	marker.id = 0;
	marker.text = "Recorded Associated Object";
	marker.pose.position.x = location.x;
	marker.pose.position.y = location.y;
	marker.pose.position.z = location.z+0.5;
	marker.scale.x = 0.1;
	marker.scale.y = 0.1;
	marker.scale.z = 0.1;
	marker.color.r = 1.0;
	marker.color.g = 0.5;
	marker.color.b = 0.0;
	marker.color.a = 1.0;
	vMarker.push_back(marker);
	marker_array.markers = vMarker;
	marker_pub.publish(marker_array);	
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
		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		float minx = x - 0.15;
		float maxx = x + 0.15;
		float miny = y - 0.20;
		float maxy = y + 0.20;
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
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*cloud_cropped, *cloud_show);
		for(int i=0; i<cloud_show->points.size(); i++)
		  {
			cloud_show->points[i].r = 255;
			cloud_show->points[i].g = 0;
			cloud_show->points[i].b = 0;
		  }
		publish_pointcloud(cloud_show);
		cout<<"point cloud pubed"<<endl;
		got_object_flag = true;
		publish_text(current_fingertip.point);	
		pcl::PLYWriter ply_saver; 
		ply_saver.write("associated_object.ply",*cloud_show);    

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
