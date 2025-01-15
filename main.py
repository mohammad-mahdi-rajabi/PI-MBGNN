from Read_training_data import read_training_test_data
from Model_Architecture import DGCNNEncoder
from Model_Training import model_training

max_time_steps = 39
delta_t = 1
list_of_folders_train = ['Pointcloud-64-P1.1', 'Pointcloud-64-P1.3' , 'Pointcloud-64-P1.5',
                           'Pointcloud-64-P1.9', 'Pointcloud-64-P2.3', 'Pointcloud-64-P2.5']
list_of_folders_val= ['Pointcloud-64-P1.7', 'Pointcloud-64-P2.1']
characteristic_lenght = 0.008
characteristic_pressure = 15

train_dataset, val_dataset = read_training_test_data(list_of_folders_train, list_of_folders_val,
                                                     max_time_steps, characteristic_lenght, characteristic_pressure, delta_t)

print("train_dataset", train_dataset)
print("val_dataset", val_dataset)

# Check

num_epochs=3
model = DGCNNEncoder(in_channels=11, out_channels=4)
model = model_training(model, train_dataset, val_dataset,  num_epochs, characteristic_lenght, delta_t)