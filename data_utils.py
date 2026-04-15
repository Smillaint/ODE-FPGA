from torchvision import datasets, transforms
from torch.utils.data import Subset

def load_stream_data(dataset_name='mnist', num_clients=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset  = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    total      = len(dataset)
    per_client = total // num_clients
    indices    = list(range(total))

    client_data = {}
    for i in range(num_clients):
        start = i * per_client
        end   = start + per_client
        client_data[i] = Subset(dataset, indices[start:end])

    return client_data, test_set

def get_stream_batch(client_dataset, round_num, speed):
    start = round_num * speed
    end   = min(start + speed, len(client_dataset))
    if start >= len(client_dataset):
        return None
    indices = list(range(start, end))
    return Subset(client_dataset.dataset,
                  [client_dataset.indices[i] for i in indices])
