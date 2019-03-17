if __name__ == '__main__':
    import torch    
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import timeit
    import pathlib
    import math

    from DilatedNetwork import Network
    from config import net_configs, batch_size, classes, num_startingpoints, saved_models_dir, dataset_dir, results_dir, process_time
    
    
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

    testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    print('running main')
    for startingpoint_index in range(num_startingpoints):
        print(f'testing for startingpoint {startingpoint_index}')

        for config in net_configs:
            try:
                print(f'loading {config["name"]}')
                current_net = Network(conv_config=config['conv_config'])
                current_net.load_state_dict(torch.load(f'{saved_models_dir}/start_{startingpoint_index}/{config["name"]}'))
                current_net.eval()

                print(f'testing {config["name"]}')
                correct = 0
                total = 0
                init_time = time.time()

                # np array containing individual batch data
                batch_times = []
                batch_results = []
                batch_labels = []

                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        batch_init_time = time.process_time() if process_time else timeit.default_timer() # start timer
                        outputs = current_net(images)
                        batch_times.extend([int(10**6 * ((time.process_time() if process_time else timeit.default_timer()) - batch_init_time)) / batch_size] * batch_size) # stop timer and save time
                        _, predicted = torch.max(outputs.data, 1)

                        total += labels.size(0)
                        correct_batch = predicted == labels
                        batch_results.extend(correct_batch)
                        batch_labels.extend(labels)
                        correct += correct_batch.sum().item()
                
                avg_time = int(10**6 * (time.time() - init_time) / total) / 1000
                successrate = round(100 * correct / total)
                
                # print results to console
                print(f'Results for network [{config["name"]}]: \n \
                    - {correct} correct out of {total} test images \n \
                    - accuracy {successrate}% \n \
                    - average time per image: {avg_time} millis')

                # save summary of data
                path = pathlib.Path(f'{results_dir}/summary.txt')
                path.parent.mkdir(parents=True, exist_ok=True)    
                with path.open('a+') as outfile:
                    outfile.write(f'sp {startingpoint_index} - {config["name"]}: {successrate}% correct, {avg_time}millis/img\n')
                        
                # save individual batch data
                np.savetxt(f'{results_dir}/{config["name"]}', np.column_stack((np.array(batch_labels), np.array(batch_results), np.array(batch_times))), fmt='%i %i %i', header='label results(bool) time(sE-6)')
                print('done writing to file')    
            
            except FileNotFoundError:
                print(f'skipping: save-file for {config["name"]} is non-existent')
            finally:
                # cleanup
                del current_net
                