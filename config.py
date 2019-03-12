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

batch_size = 4

training_epochs = 1

num_startingpoints = 1
