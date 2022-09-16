from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/home/amax/HUST_zjl/cross_view_transformers/datasets/nuscenes', verbose=True)
# nusc.list_scenes()
my_scene = nusc.scene[0]
# print(my_scene)

first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)

# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# nusc.render_sample_data(cam_front_data['token'])

# my_annotation_token = my_sample['anns'][18] # 取某个annotation的token
# nusc.render_annotation(my_annotation_token)


my_sample2 = nusc.sample[20]
# The rendering command below is commented out because it may crash in notebooks
nusc.render_sample(my_sample2['token'])
