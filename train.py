if __name__ == '__main__':

    import torch
    import torchvision
    import torchvision.transforms as transforms
    import os.path
    import pathlib
    import time

    from DilatedNetwork import Network
    from config import net_configs, batch_size, training_epochs, classes, num_startingpoints

    def train_net(net, optimizer, criterion, trainloader, epochs=1):
        print('training net')
        loss_list = []
        time_list = []

        for epoch in range(epochs):
            print(f'starting epoch {epoch + 1}')

            running_loss = 0.0
            init_time = time.time()
            for i, data in enumerate(trainloader):
                #print(f'sample {i}')
                images, labels = data

                optimizer.zero_grad()
                output = net(images)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:
                    print(f'[epoch {epoch + 1}, batch {i+1}] mean loss = {running_loss / 1000}, time = {int((time.time() - init_time) * 1000)} millis')
                    #print(f'{output[0].data}')
                    loss_list.append(int(running_loss)/1000)
                    running_loss = 0.0
                    time_list.append(int( (time.time() - init_time) * 1000))

        return (loss_list, time_list)
    
    # datasets
    print('loading datasets')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # main loop
    print('running main')
    for startingpoint_index in range(num_startingpoints):
        print(f'training configurations with startingpoint {startingpoint_index}')

        for config in net_configs:
            print(config)
            
            # skip current config if save file already exists
            if os.path.isfile(f'./cifar10saves/start_{startingpoint_index}/{config["name"]}'):
                print(f'skipping: the save for {config["name"]} already exists')
                continue
                                
            # create new net, load startingpoint and create corresponding optimizer using config
            current_net = Network(conv_config=config['conv_config'])
            #try:
            #    current_net.load_state_dict(torch.load(f'./cifar10startingpoints/start_{startingpoint_index}'))
            #except FileNotFoundError:
            #    print(f'skipping: startingpoint {startingpoint_index} not found')
            #    continue
            current_optimizer = torch.optim.Adam(current_net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
            
            # train net
            loss_list, time_list = train_net(net=current_net, optimizer=current_optimizer, epochs=training_epochs, criterion=criterion, trainloader=trainloader)
            
            # save loss history and time from training
            path = pathlib.Path(f'./cifar10training/start_{startingpoint_index}/{config["name"]}')
            path.mkdir(parents=True, exist_ok=True)
            
            with path.joinpath('loss.txt').open('w+') as outfile:
                for running_loss in loss_list:
                    outfile.write(str(running_loss) + '\n')
                print('done writing loss to file')
            
            with path.joinpath('time.txt').open('w+') as outfile:
                for running_time in time_list:
                    outfile.write(str(running_time) + '\n')
                print('done writing time to file')
            
            # save state_dict of net
            path = pathlib.Path(f'./cifar10saves/start_{startingpoint_index}')
            path.mkdir(parents=True, exist_ok=True)
            torch.save(current_net.state_dict(), f'./cifar10saves/start_{startingpoint_index}/{config["name"]}')
            print('saved state dict')
            
            # cleanup
            del loss_list
            del current_net
            del current_optimizer
        
                