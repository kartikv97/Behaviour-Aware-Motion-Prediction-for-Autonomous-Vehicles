
from model import Model
from dataProcess import getDirectFrameDict
from Grip import my_load_model, run_test


frames_list = [[200, 10001, 4, 429.917, 146.423, 37.462, 0.841, 0.632, 1.206, -0.051],
                [200, 10003, 1, 363.551, 127.567, 37.56, 1.668, 1.613, 0.773, 0.001]]

dev = 'cuda:0'
# new_dict = getDirectFrameDict(frames_list)
# print(new_dict)run_test(model)


data_root = '/home/kartik/Documents/CMSC818B/FinalProject/Behaviour-Aware-Motion-Prediction-for-Autonomous-Vehicles/GripPredModel/'

graph_args={'max_hop':2, 'num_node':120}
model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
model.to(dev)

pretrained_model_path = data_root+ 'models/model_epoch_0049.pt'
model = my_load_model(model, pretrained_model_path)

predictions = run_test(model,frames_list,graph_args)

print(predictions)