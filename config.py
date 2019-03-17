classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net_configs = [
        {
            'name': 'd1',
            'conv_config': (
				{ 'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1 },
				{ 'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1 },
				{ 'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1 },
			)
        },
		{
            'name': 'd2',
            'conv_config': (
				{ 'kernel_size': 4, 'stride': 1, 'padding': 3, 'dilation': 2 },
				{ 'kernel_size': 4, 'stride': 1, 'padding': 3, 'dilation': 2 },
				{ 'kernel_size': 4, 'stride': 1, 'padding': 3, 'dilation': 2 },
			)
        },
		{
            'name': 'd3',
            'conv_config': (
				{ 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 3 },
				{ 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 3 },
				{ 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 3 },
			)
        },
    ]

batch_size = 20

training_epochs = 1

num_startingpoints = 1

data_dir = './data'
dataset_dir = data_dir + '/dataset'
saved_models_dir = data_dir + '/saved_models'
training_data_dir = data_dir + '/traning_data'
results_dir = data_dir + '/results'

# analysis
reference_sample = 'd1'
process_time = False
plot_interval = ()