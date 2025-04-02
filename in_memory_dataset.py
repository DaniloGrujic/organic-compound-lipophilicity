import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

# From: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter, dataset)
        self.load(self.processed_paths[0])
        self.dataset = dataset
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'Lipophilicity_final.csv'

    @property
    def processed_file_names(self):
        return 'data.dt'

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.

        ############# Added this part ###############
        data_list = []
        for i, smile in enumerate(self.dataset['smiles']):
          g = from_smiles(smile)
          g.x = g.x.float()
          y = torch.tensor(self.dataset['exp'][i], dtype=torch.float).view(1, -1)
          g.y = y
          data_list.append(g)
        #############################################

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])