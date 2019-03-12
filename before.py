if __name__ == '__main__':

    import torch
    import os.path
    import pathlib

    from DilatedNetwork import Network
    from config import net_configs, batch_size, classes, num_startingpoints

    print(f'generating {num_startingpoints} starting points')
    
    path = pathlib.Path('./cifar10startingpoints')
    path.mkdir(parents=True, exist_ok=True)
    for i in range(num_startingpoints):
        current_net = Network()
        torch.save(current_net.state_dict(), f'./cifar10startingpoints/start_{i}')
    
    print('done')