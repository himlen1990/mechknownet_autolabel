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
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/registration/icp.h>
#include <mechknownet_autolabel/mrcnn.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
// ---------------only for the activated object is handheld object case-----------------

using namespace std;
using namespace cv;
typedef pcl::PointXYZ PCType;

class trajectory_demonstration
{
public:
  ros::NodeHandle nh_;

  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime< sensor_msgs::Image, sensor_msgs::Image> >* sync_input_2_; 
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub;

  ros::Subscriber camera_info_sub;
  boost::array<double,9> camera_info;
  ros::Publisher pointcloud_pub;  
  ros::Publisher pointcloud_pub_label_cloud;  
  ros::Publisher marker_pub;
  ros::Publisher marker_text_pub;  
  //ros::Publisher associate_object_cloud_pub;  
  ros::Subscriber hand_position_sub;
  ros::Subscriber control_signal_sub;


  float fx;
  float fy;
  float cx;
  float cy;
  float invfx;
  float invfy;
  tf::TransformListener listener;
  string tf_reference_frame;
  string tf_camera_frame;
  geometry_msgs::PointStamped current_hand;
  pcl::PointCloud<PCType>::Ptr current_cloud;

  pcl::PointCloud<PCType>::Ptr object1_from_mrcnn;
  pcl::PointCloud<PCType>::Ptr object2_from_mrcnn;
  pcl::PointCloud<PCType>::Ptr activated_object;
  pcl::PointCloud<PCType>::Ptr associated_object;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr labelled_cloud;

  std::vector<pcl::PointCloud<PCType>::Ptr> activated_object_trajectory;
  std::vector<std::vector<float> > pose_trajectory;
  std::vector<Eigen::Matrix4d> trans_matrix_traj;

  Eigen::Vector4f activated_object_Centroid;
  Eigen::Vector4f associated_object_Centroid;

  bool got_object_flag;
  bool got_function_part_flag;
  double timer_start;
  std::vector<geometry_msgs::Point> traj;

  float start_position_z;

  mechknownet_autolabel::mrcnn mrcnn_srv;
  ros::ServiceClient mrcnn_client;

  //string objects[3] = {"","saw","eyeglass"};
  string objects[3] = {"","roller","paper"};
  string object1;
  string object2;
  bool found_obj1_flag;
  bool found_obj2_flag;
  bool got_object_pair_flag;
  bool got_trajectory_flag;

  image_transport::ImageTransport it_;
  image_transport::Publisher image_pub;

  trajectory_demonstration():it_(nh_),labelled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>)//:camera_info{0}
  {
    rgb_sub.subscribe(nh_, "/hsrb/head_rgbd_sensor/rgb/image_rect_color", 1);
    depth_sub.subscribe(nh_, "/hsrb/head_rgbd_sensor/depth_registered/image", 1);
    sync_input_2_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime< sensor_msgs::Image,sensor_msgs::Image> >(5);//queue size
    sync_input_2_->connectInput(rgb_sub,depth_sub);
    sync_input_2_->registerCallback(boost::bind(&trajectory_demonstration::callback, this, _1, _2));
    
    camera_info_sub = nh_.subscribe("/hsrb/head_rgbd_sensor/rgb/camera_info",1, &trajectory_demonstration::camera_info_cb,this);

    hand_position_sub = nh_.subscribe("/mechknownet/hand_position", 3, &trajectory_demonstration::hand_position_callback,this);

    control_signal_sub = nh_.subscribe("/mechknownet/control_signal", 10, &trajectory_demonstration::signal_callback,this);

    pointcloud_pub = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/mechknownet/object", 1);

	pointcloud_pub_label_cloud = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("mechknownettrajoutput", 1);

	mrcnn_client = nh_.serviceClient<mechknownet_autolabel::mrcnn>("mechknownet_label_mrcnn");

	image_pub = it_.advertise("/mechknownet/image_mask", 1);

	marker_pub = nh_.advertise<visualization_msgs::MarkerArray>("mechknownet_traj_marker", 1);

	marker_text_pub = nh_.advertise<visualization_msgs::MarkerArray>("mechknownet_function_text_marker", 1);

	tf_reference_frame = "/base_link";
	tf_camera_frame = "/head_rgbd_sensor_rgb_frame";
	got_object_flag = false;
	got_function_part_flag = false;
	timer_start = -1;
	start_position_z = -1;
	object1 = "roller";//"saw";
	object2 = "paper";//"eyeglass";
	found_obj1_flag = false;
	found_obj2_flag = false;
	got_trajectory_flag = false;
#if 1 
	pcl::PLYReader Reader;
	Reader.read("./roller_init/roller.ply", *labelled_cloud);
	if(labelled_cloud->points.size() > 100)
	  cout<<"!!!!!!!!!!!!!!!!! load initply"<<endl;
	else
	  cout<<"!!!!!!!!!!!!!!!!! did not find initply"<<endl;
#endif
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

  void publish_labelled_cloud_trajectory(std::vector<Eigen::Matrix4d> traj)
  {
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
	transformation_matrix(0,3) = activated_object_Centroid(0);
	transformation_matrix(1,3) = activated_object_Centroid(1);
	transformation_matrix(2,3) = activated_object_Centroid(2);
	pcl::transformPointCloud (*labelled_cloud, *labelled_cloud, transformation_matrix);
	pcl::PointCloud<pcl::PointXYZ>::Ptr labelled_cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*labelled_cloud, *labelled_cloud_xyz);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(activated_object_trajectory[0]);
	icp.setMaximumIterations(10);
	icp.setInputTarget(labelled_cloud_xyz);
	icp.align(*labelled_cloud_xyz);
	transformation_matrix = icp.getFinalTransformation ().cast<double>();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::transformPointCloud (*labelled_cloud, *cloud_trans, transformation_matrix);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combine (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*cloud_trans, *cloud_combine);
	for(int i=1; i< traj.size(); i=i+2)
	  {
		pcl::transformPointCloud (*cloud_trans, *temp_cloud, traj[i]);
		*cloud_combine += *temp_cloud;
	  }

#if 1
	sensor_msgs::PointCloud2 pc2;
	pcl::PCLPointCloud2::Ptr pcl_pc_2(new pcl::PCLPointCloud2());
	pcl::toPCLPointCloud2 (*cloud_combine, *pcl_pc_2);
	pcl_conversions::fromPCL( *pcl_pc_2, pc2 );
	pc2.header.frame_id = tf_reference_frame;
	pointcloud_pub_label_cloud.publish(pc2);	
#endif

  }

  void publish_pose_markers(std::vector<std::vector<float> >poses, bool add_marker=true)
  {
#if 1
	visualization_msgs::MarkerArray  marker_array;
	std::vector<visualization_msgs::Marker> vMarker;
	for(int i=0; i<poses.size(); i=i+2)
	  {
		visualization_msgs::Marker marker;
		marker.header.frame_id = tf_reference_frame;
		marker.header.stamp = ros::Time::now();
		marker.type = visualization_msgs::Marker::MESH_RESOURCE;
		marker.mesh_resource = "package://mechknownet_autolabel/arrow.dae";		
		if (add_marker)
		  marker.action = visualization_msgs::Marker::ADD;
		else
		  marker.action = visualization_msgs::Marker::DELETE;
		marker.id = i;
		marker.pose.position.x = poses[i][0];
		marker.pose.position.y = poses[i][1];
		marker.pose.position.z = poses[i][2];
		marker.pose.orientation.w = poses[i][3];
		marker.pose.orientation.x = poses[i][4];
		marker.pose.orientation.y = poses[i][5];
		marker.pose.orientation.z = poses[i][6];
		marker.scale.x = 0.02;
		marker.scale.y = 0.02;
		marker.scale.z = 0.02;
	
		marker.color.r = 0.0f;
		marker.color.g = 1.0f;
		marker.color.b = 1.0f;
		marker.color.a = 1.0;
		vMarker.push_back(marker);
	  }
	marker_array.markers = vMarker;
	marker_pub.publish(marker_array);
#endif
  }

  void publish_text(std::vector<float> location)
  {
	visualization_msgs::MarkerArray  marker_array;
	std::vector<visualization_msgs::Marker> vMarker;
	visualization_msgs::Marker marker;
	marker.header.frame_id = "base_link";
	marker.header.stamp = ros::Time::now();
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::Marker::ADD;
	marker.id = 0;
	marker.text = "Learned Function: Cleaning";
	marker.pose.position.x = location[0];
	marker.pose.position.y = location[1];
	marker.pose.position.z = location[2]+0.5;
	marker.scale.x = 0.1;
	marker.scale.y = 0.1;
	marker.scale.z = 0.1;
	marker.color.r = 1.0;
	marker.color.g = 0.5;
	marker.color.b = 0.0;
	marker.color.a = 1.0;
	vMarker.push_back(marker);
	marker_array.markers = vMarker;
	marker_text_pub.publish(marker_array);	
  }

  
  void find_out_associated_object()
  {

	Eigen::Vector4f obj1Centroid;
	Eigen::Vector4f obj2Centroid;
	pcl::compute3DCentroid(*object1_from_mrcnn, obj1Centroid);
	pcl::compute3DCentroid(*object2_from_mrcnn, obj2Centroid);
	float x1 = obj1Centroid(0);
	float y1 = obj1Centroid(1);
	float z1 = obj1Centroid(2);
	float dis1 = abs(x1-current_hand.point.x)*abs(x1-current_hand.point.x)
	  +abs(y1-current_hand.point.y)*abs(y1-current_hand.point.y)
	  +abs(z1-current_hand.point.z)*abs(z1-current_hand.point.z);
	float dis_sqrt1 = sqrt(dis1);
	float x2 = obj2Centroid(0);
	float y2 = obj2Centroid(1);
	float z2 = obj2Centroid(2);
	float dis2 = abs(x2-current_hand.point.x)*abs(x2-current_hand.point.x)
	  +abs(y2-current_hand.point.y)*abs(y2-current_hand.point.y)
	  +abs(z2-current_hand.point.z)*abs(z2-current_hand.point.z);
	float dis_sqrt2 = sqrt(dis2);
	if (dis_sqrt1 > dis_sqrt2)
	  {
		associated_object = object1_from_mrcnn;
		associated_object_Centroid = obj1Centroid;
		activated_object_Centroid = obj2Centroid;
	  }
	else
	  {
		associated_object = object2_from_mrcnn;
		associated_object_Centroid = obj2Centroid;
		activated_object_Centroid = obj1Centroid;
	  }
  }


  void signal_callback(const std_msgs::String signal)
  {

	string data = signal.data;
	cout<<"??????????????"<<data<<endl;
	if(data=="hand_stable" && found_obj1_flag && found_obj2_flag)
	  {
		cout<<"!!!!!!!!!!!!!!"<<current_hand.point<<endl;
		float x = current_hand.point.x;
		float y = current_hand.point.y;
		float z = current_hand.point.z;
		find_out_associated_object();
		start_position_z = z;
		//crop bounding box around the hand
		float minx = x - 0.5;//0.3
		float maxx = x + 0.1;
		float miny = y - 0.2;
		float maxy = y + 0.2;
		float minz = z - 0.1;
		float maxz = z + 0.1;

		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		pcl::CropBox<PCType> boxFilter;

		boxFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
		boxFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));
		boxFilter.setInputCloud(current_cloud);
		boxFilter.filter(*cloud_cropped);
		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_cropped, *cloud_cropped, nan_indices);		
		
		//remove desk plane 
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		// Optional
		Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0); 
		seg.setAxis(axis);
		seg.setEpsAngle(10.0*(3.14/180.0f));
		seg.setOptimizeCoefficients (true);
		// Mandatory
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setDistanceThreshold (0.01);

		seg.setInputCloud (cloud_cropped);
		seg.segment (*inliers, *coefficients);


		pcl::PointCloud <PCType>::Ptr cloud_no_plane (new pcl::PointCloud <PCType>);
		pcl::ExtractIndices<PCType> extract;
		extract.setInputCloud(cloud_cropped);
		extract.setIndices(inliers);
		extract.setNegative(true);
		extract.filter(*cloud_no_plane);

		if (inliers->indices.size () == 0)
		  {
			PCL_ERROR ("Could not estimate a planar model for the given dataset.");
		  }
		//end
		activated_object = cloud_no_plane;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr associate_cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*cloud_no_plane, *cloud_show);
		pcl::copyPointCloud(*associated_object, *associate_cloud_show);

		for(int i=0; i<cloud_show->points.size(); i++)
		  {
			cloud_show->points[i].r = 255;
			cloud_show->points[i].g = 80;
			cloud_show->points[i].b = 80;
		  }
		for(int i=0; i<associate_cloud_show->points.size(); i++)
		  {
			associate_cloud_show->points[i].r = 0;
			associate_cloud_show->points[i].g = 255;
			associate_cloud_show->points[i].b = 0;
		  }
		*cloud_show += *associate_cloud_show;
		publish_pointcloud(cloud_show);
		cout<<"point cloud pubed"<<endl;
		got_object_flag = true;
		//pcl::PLYWriter ply_saver; 
		//ply_saver.write("hand_object_cloud.ply",*cloud_show);    
		activated_object_trajectory.push_back(activated_object);
	  }
	if(data=="hand_moving" && !got_trajectory_flag)
	  {
		cout<<"moving!!!!!!!"<<current_hand.point<<endl;
		float x = current_hand.point.x;
		float y = current_hand.point.y;
		float z = current_hand.point.z;

		float minx = x - 0.3;
		float maxx = x;
		float miny = y - 0.2;
		float maxy = y + 0.2;
		float minz = z - 0.1;
		float maxz = z + 0.1;

		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		pcl::CropBox<PCType> boxFilter;

		boxFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
		boxFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));
		boxFilter.setInputCloud(current_cloud);
		boxFilter.filter(*cloud_cropped);
		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_cropped, *cloud_cropped, nan_indices);		

#if 0		
		if (z - start_position_z < 0.1)
		  {
			//remove desk plane 
			pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
			// Create the segmentation object
			pcl::SACSegmentation<pcl::PointXYZ> seg;
			// Optional
			Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0); 
			seg.setAxis(axis);
			seg.setEpsAngle(10.0*(3.14/180.0f));
			seg.setOptimizeCoefficients (true);
			// Mandatory
			seg.setModelType (pcl::SACMODEL_PLANE);
			seg.setMethodType (pcl::SAC_RANSAC);
			seg.setDistanceThreshold (0.01);
			seg.setInputCloud (cloud_cropped);
			seg.segment (*inliers, *coefficients);

			pcl::PointCloud <PCType>::Ptr cloud_no_plane (new pcl::PointCloud <PCType>);
			pcl::ExtractIndices<PCType> extract;
			extract.setInputCloud(cloud_cropped);
			extract.setIndices(inliers);
			extract.setNegative(true);
			extract.filter(*cloud_no_plane);
			
			if (inliers->indices.size () == 0)
			  {
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
			  }
			//end

			activated_object = cloud_no_plane;
		  }
		else
		  activated_object = cloud_cropped;
#endif
		//remove_table_top
	if(true)
	  {
		pcl::search::Search <PCType>::Ptr tree (new pcl::search::KdTree<PCType>);
		pcl::PointCloud <PCType>::Ptr cloud_cropped_no_table (new pcl::PointCloud <PCType>);
		pcl::PointCloud <PCType>::Ptr cloud_removed_outliers (new pcl::PointCloud <PCType>);
		pcl::ConditionAnd<PCType>::Ptr range_cond (new pcl::ConditionAnd<PCType> ());
		//range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::GT, 0.75)));//for hsr81
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::GT, 0.76)));//for hsr43
		pcl::ConditionalRemoval<PCType> condrem;
		condrem.setCondition (range_cond);
		condrem.setInputCloud (cloud_cropped);
		condrem.setKeepOrganized(true);
		condrem.filter (*cloud_cropped_no_table);

		//remove outliers
		pcl::RadiusOutlierRemoval<PCType> outrem;
		outrem.setInputCloud(cloud_cropped_no_table);
		outrem.setRadiusSearch(0.8);
		outrem.setMinNeighborsInRadius (2);
		outrem.setKeepOrganized(true);
		outrem.filter (*cloud_cropped_no_table);
		
		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_cropped_no_table, *cloud_cropped_no_table, nan_indices);
		activated_object = cloud_cropped_no_table;
	  }

		//end remove_table_top
	if (activated_object_trajectory.size()<15) //18
	  {
		activated_object_trajectory.push_back(activated_object);
	  }

	else
	  {
		//publish_labelled_cloud_trajectory(pose_trajectory);
		//icp
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(activated_object_trajectory[0]);
		icp.setMaximumIterations(10);
		std::vector<float> start_pose{activated_object_Centroid(0),activated_object_Centroid(1),activated_object_Centroid(2),1.0,1.0,0.0,0.0};// x -> 1.0
		cout<<"x "<<start_pose[0]<<" y "<<start_pose[1]<<" z "<<start_pose[2]<<endl;
			
		//compute 6D poses
		pose_trajectory.push_back(start_pose);
		for(int i=1; i<activated_object_trajectory.size(); i++)
		  {
			icp.setInputTarget(activated_object_trajectory[i]);
			pcl::PointCloud<pcl::PointXYZ> temp_cloud;
			icp.align(temp_cloud);
			Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation ().cast<double>();
			std::vector<float> transformed_pose = pose_transform(transformation_matrix,start_pose);
			cout<<"x "<<transformed_pose[0]<<" y "<<transformed_pose[1]<<" z "<<transformed_pose[2]<<" qw "<<transformed_pose[3]<<" qx "<<transformed_pose[4]<<" qy "<<transformed_pose[5]<<" qz "<<transformed_pose[6]<<endl;

			trans_matrix_traj.push_back(transformation_matrix);
			pose_trajectory.push_back(transformed_pose);
			got_trajectory_flag = true;
		  }
			publish_pose_markers(pose_trajectory);
			publish_labelled_cloud_trajectory(trans_matrix_traj);
			publish_text(pose_trajectory[0]);

#if 0		
		cout<<"find middle pose"<<endl;
		std::vector<std::vector<float> > pose_trajectory_final;
		std::vector<float> start = pose_trajectory[0];
		std::vector<float> end = pose_trajectory[14]; //notice
		pose_trajectory_final.push_back(start);
		pose_trajectory_final.push_back(end);
		float max_dis = 0;
		int index = -1;
		for(int i=5; i<pose_trajectory.size()-1; i++)
		  {
			float dis = point_to_line_dis(start, end, pose_trajectory[i]);
			if (dis>max_dis)
			  {
				max_dis = dis;
				index = i;
			  }
			cout<<"!!!!!!!!!dis!!!!!!!!!!!!! "<<dis<<endl;
		  }
		cout<<"!!!!!!!!!max_dis!!!!!!!!!!!!! "<<max_dis<<endl;
		pose_trajectory_final.push_back(pose_trajectory[index]);
		//publish_pose_markers(pose_trajectory,false);
		//publish_pose_markers(pose_trajectory_final);
#endif		

		pcl::PLYWriter ply_saver; 
		char buffer[30];			
		for(int i=0; i<activated_object_trajectory.size(); i++)
		  {
			snprintf(buffer, sizeof(buffer), "trajectory/frame%04d.ply", i);
			std::string path = buffer;
			ply_saver.write(path,*activated_object_trajectory[i]);    
		  }

	  }

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr associate_cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*activated_object, *cloud_show);
		pcl::copyPointCloud(*associated_object, *associate_cloud_show);
		for(int i=0; i<cloud_show->points.size(); i++)
		  {
			cloud_show->points[i].r = 255;
			cloud_show->points[i].g = 80;
			cloud_show->points[i].b = 80;
		  }
		for(int i=0; i<associate_cloud_show->points.size(); i++)
		  {
			associate_cloud_show->points[i].r = 0;
			associate_cloud_show->points[i].g = 255;
			associate_cloud_show->points[i].b = 0;
		  }
		*cloud_show += *associate_cloud_show;

		publish_pointcloud(cloud_show);
		cout<<"point cloud pubed"<<endl;

	  }
  }

  void hand_position_callback(const geometry_msgs::PointStamped point)
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

	listener.transformPoint(tf_reference_frame, pt, pt_transformed);
	current_hand = pt_transformed;
  }
  

  void callback(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD)
  {

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImage cv_img;
    try
      {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB, sensor_msgs::image_encodings::BGR8);
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

    pcl::PointCloud<PCType>::Ptr cloud = depth_to_pointcloud(depth_f);
	cloud->header.frame_id = cv_ptrD->header.frame_id;

	try{
	  listener.waitForTransform(tf_reference_frame, "/head_rgbd_sensor_rgb_frame", ros::Time(0), ros::Duration(3.0));
	}	   
	catch(tf::TransformException ex)
	  {
		ROS_ERROR("%s",ex.what());
	  }

	pcl::PointCloud<PCType>::Ptr cloud_trans(new pcl::PointCloud<PCType>());
	pcl_ros::transformPointCloud(tf_reference_frame, *cloud, *cloud_trans, listener);

	current_cloud = cloud_trans;


	if (!got_object_pair_flag) //call mrcnn to get object-pair at the very beginning
	  {
		sensor_msgs::Image rgbimg = *msgRGB;
		mrcnn_srv.request.Image = rgbimg;
		if(mrcnn_client.call(mrcnn_srv))
		  {
			std::vector<int32_t> cls_ids = mrcnn_srv.response.Class_ids;
			std::vector<int32_t> scores = mrcnn_srv.response.Scores;
			std::vector<sensor_msgs::Image> masks = mrcnn_srv.response.Masks;
			if(cls_ids.size()>0)
			  {
				show_mask_image(masks,RGBImg);
				cout<<"I found "<<cls_ids.size()<<" objects"<<endl;
				for(int i=0; i<cls_ids.size(); i++) 
				  {
					cout<<"I found "<<objects[cls_ids[i]]<<endl;
					if(objects[cls_ids[i]] == object1)
					  {
						
						ROS_INFO("found object1");
						found_obj1_flag = true;
						object1_from_mrcnn = mask_to_pointcloud(masks[i],depth_f);
						
					  }
					else
					  ROS_INFO("fail to detect object1");
					if(objects[cls_ids[i]] == object2)
					  {
						
						ROS_INFO("found object2");
						found_obj2_flag = true;
						object2_from_mrcnn = mask_to_pointcloud(masks[i],depth_f);
					  }
				  }
			  }
			if(found_obj1_flag && found_obj2_flag)
			  {
				pcl::PointCloud<PCType>::Ptr obj_pair_cloud (new pcl::PointCloud<PCType>);
				pcl::copyPointCloud(*object1_from_mrcnn, *obj_pair_cloud);
				*obj_pair_cloud += *object2_from_mrcnn;

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_pair_cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);
				pcl::copyPointCloud(*obj_pair_cloud, *obj_pair_cloud_show);
				for(int i=0; i<obj_pair_cloud_show->points.size(); i++)
				  {
					obj_pair_cloud_show->points[i].r = 160;
					obj_pair_cloud_show->points[i].g = 160;
					obj_pair_cloud_show->points[i].b = 255;
				  }
				publish_pointcloud(obj_pair_cloud_show);
				cout<<"object pair point cloud pubed"<<endl;
				cout<<"OKKKKKKKKKKKKKKK"<<endl;
				//got_object_pair_flag=true; // keep detecting for gpu device
			  }
			got_object_pair_flag=true;//only detect once for cpu device
		  }
	  }

  } //end void callback

  void show_mask_image(std::vector<sensor_msgs::Image> masks, cv::Mat RGBImg)
  {

	cv_bridge::CvImageConstPtr cv_ptrMONO;
	std::vector<cv::Mat> maskImgs;
	for(int i=0; i<masks.size(); i++)
	  {
		try
		  {
			cv_ptrMONO = cv_bridge::toCvCopy(masks[i], sensor_msgs::image_encodings::MONO8);
		  }
		catch (cv_bridge::Exception& e)
		  {
			ROS_ERROR("cv_bridge exception: %s", e.what());
		  }
		cv::Mat BiImage;		
		cv_ptrMONO->image.copyTo(BiImage);
		maskImgs.push_back(BiImage);
	  }
	
	cv::Mat combine_mask = maskImgs[0];
	if(maskImgs.size()>1)
	  {
		for(int i=1; i<maskImgs.size(); i++)
		  cv::add(combine_mask,maskImgs[i] ,combine_mask);
	  }


	cv::Mat color_mask;
	bitwise_and(RGBImg,RGBImg,color_mask,combine_mask);
	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_mask).toImageMsg();
	image_pub.publish(msg);
  }


  pcl::PointCloud<PCType>::Ptr mask_to_pointcloud(sensor_msgs::Image mask, cv::Mat depthImg)
  {
	cv_bridge::CvImageConstPtr cv_ptrMONO;
	try
	  {
		cv_ptrMONO = cv_bridge::toCvCopy(mask, sensor_msgs::image_encodings::MONO8);
	  }
	catch (cv_bridge::Exception& e)
	  {
		ROS_ERROR("cv_bridge exception: %s", e.what());
	  }
	cv::Mat BiImage;
	cv_ptrMONO->image.copyTo(BiImage);
    pcl::PointCloud<PCType>::Ptr cloud = depth_to_pointcloud(depthImg, BiImage);
	pcl::PointCloud<PCType>::Ptr processed_cloud = pointcloud_processor(cloud);
	processed_cloud->header.frame_id = tf_camera_frame;
	try{
	  listener.waitForTransform(tf_reference_frame, tf_camera_frame, ros::Time(0), ros::Duration(3.0));
	}
	catch (tf::TransformException ex){
	  ROS_ERROR("%s",ex.what());
	}
	pcl::PointCloud<PCType>::Ptr cloud_trans(new pcl::PointCloud<PCType>());
	pcl_ros::transformPointCloud(tf_reference_frame, *processed_cloud, *cloud_trans, listener);

	//remove_table_top
	if(true)
	  {
		pcl::search::Search <PCType>::Ptr tree (new pcl::search::KdTree<PCType>);
		pcl::PointCloud <PCType>::Ptr cloud_cropped (new pcl::PointCloud <PCType>);
		pcl::PointCloud <PCType>::Ptr cloud_removed_outliers (new pcl::PointCloud <PCType>);
		pcl::ConditionAnd<PCType>::Ptr range_cond (new pcl::ConditionAnd<PCType> ());
		//range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::GT, 0.75)));//for hsr81
		range_cond->addComparison (pcl::FieldComparison<PCType>::ConstPtr (new pcl::FieldComparison<PCType> ("z", pcl::ComparisonOps::GT, 0.74)));//for hsr43
		pcl::ConditionalRemoval<PCType> condrem;
		condrem.setCondition (range_cond);
		condrem.setInputCloud (cloud_trans);
		condrem.setKeepOrganized(true);
		condrem.filter (*cloud_cropped);

		//remove outliers
		pcl::RadiusOutlierRemoval<PCType> outrem;
		outrem.setInputCloud(cloud_cropped);
		outrem.setRadiusSearch(0.8);
		outrem.setMinNeighborsInRadius (2);
		outrem.setKeepOrganized(true);
		outrem.filter (*cloud_cropped);
		
		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_cropped, *cloud_cropped, nan_indices);
		return cloud_cropped;
	  }

	//end remove_table_top
	return cloud_trans;

  }

  pcl::PointCloud<PCType>::Ptr depth_to_pointcloud(cv::Mat depth_f, cv::Mat mask=cv::Mat{})
  {
    pcl::PointCloud<PCType>::Ptr cloud(new pcl::PointCloud<PCType>());	
	if(mask.empty())
	  {
		for(int i = 0; i < depth_f.cols; i++) {
		  for(int j = 0; j < depth_f.rows; j++) {
			float z = depth_f.at<float>(j,i);
			if(z>0)
			  {
				pcl::PointXYZ point;
				point.x = (i-cx)*z*invfx;
				point.y = (j-cy)*z*invfy;
				point.z = z;
				cloud->points.push_back(point);
			  }    
		  }
		}
	  }
	else
	  {
		for(int i = 0; i < depth_f.cols; i++) {
		  for(int j = 0; j < depth_f.rows; j++) {
			float z = depth_f.at<float>(j,i);
			if(z>0 && mask.at<uchar>(j,i)==255)
			  {
				pcl::PointXYZ point;
				point.x = (i-cx)*z*invfx;
				point.y = (j-cy)*z*invfy;
				point.z = z;
				cloud->points.push_back(point);
			  }    
		  }
		}
	  }
	return cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_processor(pcl::PointCloud <PCType>::Ptr &cloud)
  {
	std::vector <pcl::PointIndices> clusters;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.06); // 6cm
	ec.setMinClusterSize (300);
	//ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud);
	ec.extract (clusters);
	cout<<"cloud size "<<cloud->points.size()<<endl;
	cout<<"clusters size "<<clusters.size()<<endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
	int largest_piece_index = 0;
	int cluster_size = clusters[0].indices.size();
	if(clusters.size()>1) //retrun the largest piece
	  {
		for (int i=0; i<clusters.size(); i++)
		  {
			if(clusters[i].indices.size() > cluster_size)
			  largest_piece_index = i;
		  }
	  }
	for (std::vector<int>::const_iterator pit = clusters[largest_piece_index].indices.begin (); pit != clusters[largest_piece_index].indices.end (); ++pit) 
	  cloud_cluster->points.push_back (cloud->points[*pit]); //*

	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;

	return cloud_cluster;
  }

  std::vector<float> pose_transform(Eigen::Matrix4d transformation_matrix, std::vector<float> origin_pose)
  {
	std::vector<float> pose_after_transform;
	Eigen::Quaternionf q;
	q.w() = origin_pose[3];
	q.x() = origin_pose[4];
	q.y() = origin_pose[5];
	q.z() = origin_pose[6];
	Eigen::Matrix3f RM = q.normalized().toRotationMatrix();
	Eigen::Matrix4f pose_matrix = Eigen::Matrix4f::Identity();
	pose_matrix.block<3,3>(0,0) = RM;
	pose_matrix(0,3) = origin_pose[0];
	pose_matrix(1,3) = origin_pose[1];
	pose_matrix(2,3) = origin_pose[2];

	Eigen::Matrix4f new_pose_matrix = transformation_matrix.cast<float>() * pose_matrix;
	Eigen::Quaternionf new_q(new_pose_matrix.block<3,3>(0,0));
	std::vector<float> new_pose(7,0);
	new_pose[0] = new_pose_matrix(0,3);
	new_pose[1] = new_pose_matrix(1,3);
	new_pose[2] = new_pose_matrix(2,3);
	new_pose[3] = new_q.w();
	new_pose[4] = new_q.x();
	new_pose[5] = new_q.y();
	new_pose[6] = new_q.z();

	return new_pose;
  }


  float point_to_line_dis(std::vector<float> start, std::vector<float> end, std::vector<float> mid) 
  {
	std::vector<float> es = {end[0]-start[0],end[1]-start[1],end[2]-start[2]};
	std::vector<float> ms = {mid[0]-start[0],mid[1]-start[1],mid[2]-start[2]};
	std::vector<float> ms_cross_es = {(es[1]*ms[2]-es[2]*ms[1]),(es[2]*ms[0] - es[0]*ms[2]),(es[0]*ms[1] - es[1]*ms[0])};
	float area = sqrt(pow(ms_cross_es[0], 2) + pow(ms_cross_es[1], 2) + pow(ms_cross_es[2], 2)); 
	float dis = area/sqrt(pow(es[0], 2) + pow(es[1], 2) + pow(es[2], 2)); 
	return dis;
  }


};
int main(int argc, char** argv)
{
  ros::init(argc, argv, "rgb_depth");
  trajectory_demonstration fd;
  ros::spin();

  return 0;
}
