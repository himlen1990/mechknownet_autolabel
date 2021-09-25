#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/common.h>
#include <fstream>
#include <string>

using namespace std;

class coordinate_convertor
{
public:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr activated_object;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr associated_object;
  std::vector<std::vector<float> > pose_label;
  
  coordinate_convertor():activated_object(new pcl::PointCloud<pcl::PointXYZRGB>),associated_object(new pcl::PointCloud<pcl::PointXYZRGB>)
  {
	pcl::PLYReader Reader;
	Reader.read("./result/activated_obj_traj.ply", *activated_object);	
	Reader.read("./result/associated_obj_traj.ply", *associated_object);	
	read_txt("./result/trajectory_label1.txt");
	read_txt("./result/trajectory_label2.txt");
	read_txt("./result/frame0000_grasp.txt");
	cout<<pose_label.size()<<endl;
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	//pcl::copyPointCloud(*activated_object, *combined_cloud);
	//*combined_cloud += *associated_object;


	Eigen::Affine3f transform_to_origin_ac;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_ac = cloud_transform_to_origin(activated_object,transform_to_origin_ac);

	Eigen::Affine3f transform_to_origin_as;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_as = cloud_transform_to_origin(associated_object,transform_to_origin_as);

	rotate_cloud_and_label(cloud_transformed_ac,cloud_transformed_as,transform_to_origin_ac,transform_to_origin_as);
  }
  
  void rotate_cloud_and_label(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_ac, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_as, Eigen::Affine3f transform_to_origin_ac, Eigen::Affine3f transform_to_origin_as)
  {
	for(int i=0; i<3; i++)
	//for(int i=0; i<18; i++)
	  {
		//float theta = (-3.14/1.1) + (0.1 * i);
		float theta =0.5+0.2*i;
		//float theta = 0.9*i;
		cout<<theta<<endl;
		Eigen::Affine3f rotation_mat = Eigen::Affine3f::Identity();
		rotation_mat.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr activated_obj_after_rotate (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr associated_obj_after_rotate (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::transformPointCloud (*cloud_transformed_ac, *activated_obj_after_rotate, rotation_mat);
		pcl::transformPointCloud (*cloud_transformed_as, *associated_obj_after_rotate, rotation_mat);
	
		Eigen::Affine3f pose_trans = transform_pose(pose_label[0],transform_to_origin_ac,rotation_mat);
		Eigen::Affine3f pose_trans2 = transform_pose(pose_label[1],transform_to_origin_as,rotation_mat);
		Eigen::Affine3f pose_trans3 = transform_pose(pose_label[2],transform_to_origin_as,rotation_mat);
		Eigen::Affine3f pose_trans_grasp = transform_pose(pose_label[3],transform_to_origin_ac,rotation_mat);

		Eigen::Quaternionf label_q(pose_trans.matrix().block<3,3>(0,0));
		Eigen::Quaternionf label_q2(pose_trans2.matrix().block<3,3>(0,0));
		Eigen::Quaternionf label_q3(pose_trans3.matrix().block<3,3>(0,0));
		Eigen::Quaternionf label_q_grasp(pose_trans_grasp.matrix().block<3,3>(0,0));

		char buffer[30];
		snprintf(buffer, sizeof(buffer), "frame%04d.ply", i);
		std::string suffix = buffer;
		std::string ply_path= "result/ac/" + suffix;

		pcl::PLYWriter ply_saver;	
		ply_saver.write(ply_path.c_str(),*activated_obj_after_rotate);

		snprintf(buffer, sizeof(buffer), "frame%04d.ply", i);
		suffix = buffer;
		ply_path= "result/as/" + suffix;


		ply_saver.write(ply_path.c_str(),*associated_obj_after_rotate);
		
		snprintf(buffer, sizeof(buffer), "frame%04d.txt", i);
		suffix = buffer;
		std::string txt_path= "result/ac/" + suffix;

		std::ofstream file;
#if 1
		file.open(txt_path.c_str());
		file<<pose_trans.translation()(0)<<" "<<pose_trans.translation()(1)<<" "<<pose_trans.translation()(2)<<" "<<label_q.w()<<" "<<label_q.x()<<" "<<label_q.y()<<" "<<label_q.z()<<std::endl;
		file.close();
#endif
		snprintf(buffer, sizeof(buffer), "frame%04d.txt", i);
		suffix = buffer;
		txt_path= "result/as/" + suffix;

		file.open(txt_path.c_str());
		file<<pose_trans2.translation()(0)<<" "<<pose_trans2.translation()(1)<<" "<<pose_trans2.translation()(2)<<" "<<label_q2.w()<<" "<<label_q2.x()<<" "<<label_q2.y()<<" "<<label_q2.z()<<std::endl;
		file<<pose_trans3.translation()(0)<<" "<<pose_trans3.translation()(1)<<" "<<pose_trans3.translation()(2)<<" "<<label_q3.w()<<" "<<label_q3.x()<<" "<<label_q3.y()<<" "<<label_q3.z()<<std::endl;
		file.close();

		snprintf(buffer, sizeof(buffer), "frame%04d_grasp.txt", i);
		//snprintf(buffer, sizeof(buffer), "frame%04d.txt", i);
		suffix = buffer;
		txt_path= "result/ac/" + suffix;

		file.open(txt_path.c_str());
		file<<pose_trans_grasp.translation()(0)<<" "<<pose_trans_grasp.translation()(1)<<" "<<pose_trans_grasp.translation()(2)<<" "<<label_q_grasp.w()<<" "<<label_q_grasp.x()<<" "<<label_q_grasp.y()<<" "<<label_q_grasp.z()<<std::endl;
		file.close();
	  }
  }

  Eigen::Affine3f transform_pose(std::vector<float> pose, Eigen::Affine3f transform_to_origin, Eigen::Affine3f rotation)
  {
	Eigen::Quaternionf quater;
	quater.w() = pose[3];
	quater.x() = pose[4];
	quater.y() = pose[5];
	quater.z() = pose[6];

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << pose[0],pose[1],pose[2];
  transform.matrix().block<3,3>(0,0) = quater.toRotationMatrix();

  transform = transform_to_origin*transform;
  transform =  rotation*transform;  

  return transform;
  }


  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transform_to_origin(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Affine3f& transform_mat)
  {
  Eigen::Vector4f pcaCentroid;
  pcl::compute3DCentroid(*cloud, pcaCentroid);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans(new pcl::PointCloud<pcl::PointXYZRGB>);

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << -pcaCentroid(0), -pcaCentroid(1), -pcaCentroid(2);
  pcl::transformPointCloud (*cloud, *cloud_trans, transform);
  transform_mat = transform;

  return cloud_trans;
  }


  void read_txt(std::string filename)
  {
	std::ifstream file;
	file.open(filename,std::ios_base::in);
	std::string line;
	while (std::getline(file, line))
	  {
		std::stringstream ss(line);
		std::vector <float> element;
		while (getline(ss,line,' '))
		  {
			element.push_back(std::atof(line.c_str()));
		  }
		pose_label.push_back(element);
	  }	
  }


};

int main(int argc, char** argv)
{
  coordinate_convertor cc;
  return 0;
}
