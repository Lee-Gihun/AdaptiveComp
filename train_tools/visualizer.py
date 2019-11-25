from sklearn.manifold.t_sne import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
from .utils import directory_setter, path_setter
import torch, os

__all__ = ['VisualizationHandler']


class VisualizationHandler():
    """
    Inspector for learned network. Conducts following inspections:

    """
    def __init__(self, Network, dataloaders, dataset_sizes, num_classes=None, num_path=4, device='cuda:0', 
                 phase='test', save_path='./results/inspection', name='test'):
        self.Network = Network.to(device).eval()
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.phase = phase
        self.num_path = num_path
        self.save_path = save_path
        self.name = name
        
        self.features = [None] * num_path
        self.correct = [None] * num_path
        self.confidence = [None] * num_path
        self.embedding = [None] * num_path
        self.labels = None
        
        num_classes = num_classes if num_classes is not None else Network.num_classes
        
        if num_classes == 10:
            self.classes = [
                'airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
        
        elif num_classes == 100:
            self.classes = [
                'beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 
                'bottles', 'bowls', 'cans', 'cups', 'plates', 
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television', 
                'bed', 'chair', 'couch', 'table', 'wardrobe', 
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
                'bear', 'leopard', 'lion', 'tiger', 'wolf', 
                'bridge', 'castle', 'house', 'road', 'skyscraper', 
                'cloud', 'forest', 'mountain', 'plain', 'sea', 
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 
                'crab', 'lobster', 'snail', 'spider', 'worm', 
                'baby', 'boy', 'girl', 'man', 'woman', 
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 
                'maple', 'oak', 'palm', 'pine', 'willow', 
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
        
    def _forward(self, x):
        """
        Inference for a single batch. 
        
        [returns]   (Tensor) outputs : inference result tensor
                    (Tensor) features : features to be visualized
        """
        outputs, features, confidence = self.Network(x)
        return outputs, features, confidence
    
    
    def _prediction(self, outputs):
        """
        Prediction for a single batch inference result.
        
        [returns]   (Tensor) max_logits : maximum softmax output value tensor
                    (Tensor) pred : prediction result tensor
        """
        preds = [None] * self.num_path
        
        for i in range(self.num_path):
            _, pred = torch.max(outputs[i], 1)
            preds[i] = pred
        
        return preds
    
    
    def _attr_updater(self, tensor, path_idx=0, mode='features'):
        tensor = tensor.cpu()
        
        if mode == 'features':
            if self.features[path_idx] is None:
                self.features[path_idx] = tensor

            else:
                self.features[path_idx] = torch.cat((self.features[path_idx], tensor), dim=0)
        
        elif mode == 'labels':
            if self.labels is None:
                self.labels = tensor
            else:
                self.labels = torch.cat((self.labels, tensor), dim=0)
                
        elif mode == 'confidence':
            if self.confidence[path_idx] is None:
                self.confidence[path_idx] = tensor
            else:
                self.confidence[path_idx] = torch.cat((self.confidence[path_idx], tensor), dim=0)
                
        elif mode == 'correct':
            if self.correct[path_idx] is None:
                self.correct[path_idx] = tensor
            else:
                self.correct[path_idx] = torch.cat((self.correct[path_idx], tensor), dim=0)    
    
    def _inference(self):
        """
        Inference for 
        
        [args]      (str) phase : use test or valid set 'valid' or 'test'
        
        [returns]   

        """
        correct, count = [0] * self.num_path, [0] * self.num_path
        size = self.dataset_sizes[self.phase]
                 
        with torch.no_grad():
            for inputs, labels in self.dataloaders[self.phase]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Inference from Network
                outputs, features, confidence = self._forward(inputs)
                preds = self._prediction(outputs)
                
                self._attr_updater(labels, mode='labels')
                
                for i in range(self.num_path):
                    self._attr_updater(features[i], path_idx=i, mode='features')
                    self._attr_updater(confidence[i], path_idx=i, mode='confidence')
                    correct_tensor = (preds[i] == labels)
                    self._attr_updater(correct_tensor, path_idx=i, mode='correct')
        
        self._to_numpy()

    
    def _embedder(self, mode='umap', **kwargs):
        self._inference()
        print('conducting feature embedding...')

        if mode == 'umap':
            reducer = UMAP(**kwargs)
        
        elif mode == 'tsne':
            reducer = TSNE(**kwargs)
        
        
        for i in range(self.num_path):
            self.embedding[i] = reducer.fit_transform(self.features[i])
        
        print('features are embedded as shape {}'.format(self.embedding[-1].shape))

        
    def visualizer(self, phase='test', mode='umap', **kwargs):
        save_path = path_setter(self.save_path, self.name, 'visualization')
        directory_setter(save_path, make_dir=True)
        self._embedder(mode, **kwargs)

        
        for c in range(self.num_path):            
            plt.figure(figsize=(8, 6))
            x, y = self.embedding[c][:,0], self.embedding[c][:,1]

            for i in range(len(self.classes)):
                if len(self.classes)==100:
                    if i % 10 == 0:
                        y_i = self.labels == i
                        plt.scatter(x[y_i], y[y_i], label=self.classes[i], alpha=0.7)
                else:
                    y_i = self.labels == i
                    plt.scatter(x[y_i], y[y_i], label=self.classes[i], alpha=0.7)
                    
            plt.legend()

            fname = os.path.join(save_path, '%s_path%d_%s.png' % (self.name, c, mode))
            plt.savefig(fname)
        
        """
        explanation

        [args]      (type) name: 

        [returns]   (type) name : 
        """
        """
        visualize learned representation.
        (args) type: select visualization type between 'umap' and 'tsne'
        """
        
    def _to_numpy(self):
        self.labels = self.labels.cpu().numpy()

        for i in range(self.num_path):            
            self.features[i] = self.features[i].cpu().numpy()
            self.correct[i] = self.correct[i].cpu().numpy()
    
    def set_name(self, name):
        self.name = name