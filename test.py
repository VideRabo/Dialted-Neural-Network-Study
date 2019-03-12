if __name__ == '__main__':
    import torch    
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    from DilatedNetwork import Network
    from config import net_configs, batch_size, classes, num_startingpoints
    
    
    # utility functions
    def show_image(img):
        '''show a normalized torch tensor as an image'''
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()     # convert tensor to ndarray
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # transpose dimensions

    # datasets
    print('loading datasets')    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    print('running main')
    for startingpoint_index in range(num_startingpoints):
        print(f'testing for startingpoint {startingpoint_index}')

        for config in net_configs:
            try:
                print(f'loading {config["name"]}')
                current_net = Network(conv_config=config['conv_config'])
                current_net.load_state_dict(torch.load(f'./cifar10saves/start_{startingpoint_index}/{config["name"]}'))
                current_net.eval()

                print(f'testing {config["name"]}')
                correct = 0
                total = 0
                init_time = time.time()
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = current_net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                avg_time = int(10**6 * (time.time() - init_time) / total) / 1000
                successrate = int(100 * correct / total)
                
                print(f'Accuracy of network [ {config["name"]} ] on {total} test images: {successrate}%, average time per image: {avg_time} millis')
                with open('./cifar10result.txt', 'a+') as outfile:
                    outfile.write(f'sp {startingpoint_index} - {config["name"]}: {successrate}% correct, {avg_time}millis/img\n')
                
                print(f'done testing {config["name"]}')
            except FileNotFoundError:
                print(f'skipping: save-file for {config["name"]} is non-existent')
            finally:
                # cleanup
                del current_net
                