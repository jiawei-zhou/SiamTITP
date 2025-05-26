from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .satsot import SatSOTDataset
from .sv248s import SV248SDataset
from .viso import VISODataset
from .ootb import OOTBDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' == name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'SatSOT' == name:
            dataset = SatSOTDataset(**kwargs)
        elif 'SV248S' == name:
            dataset = SV248SDataset(**kwargs)
        elif 'VISO' == name:
            dataset = VISODataset(**kwargs)
        elif 'OOTB' == name:
            dataset = OOTBDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
