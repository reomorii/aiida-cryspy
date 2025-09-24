import pickle
import pandas as pd
import io
from aiida.plugins import DataFactory

SinglefileData = DataFactory('core.singlefile')


class EAData(SinglefileData):
    """CrySPY ea_data
    
    """
    def __init__(self, ea_data, **kwargs):
        """
        SinglefileData requests file in __init__(), so ea_data must not be None.
        """
        self._internal_validate(ea_data)
        
        content = pickle.dumps(ea_data)
        handle = io.BytesIO(content)
        super().__init__(file=handle, **kwargs)
        
        
    def _internal_validate(self, ea_data):
        if len(ea_data) != 5:
            raise TypeError('size of ea_data must be 5.')
        # ea_data[0] is gen
        if not isinstance(ea_data[0], int):
            raise TypeError('ea_data[0] must be int')
        # ea_data[1] is elite_struc
        if ea_data[1] is not None:
            if not isinstance(ea_data[1], dict):
                raise TypeError('ea_data[1] must be dict')
        # ea_data[2] is elite_fitness
        if ea_data[2] is not None:
            if not isinstance(ea_data[2], dict):
                raise TypeError('ea_data[2] must be dict')
        # ea_data[3] is ea_info
        if ea_data[3] is not None:
            if not isinstance(ea_data[3], pd.DataFrame):
                raise TypeError('ea_data[3] must be pd.DataFrame')
        # ea_data[4] is ea_origin
        if ea_data[4] is not None:
            if not isinstance(ea_data[4], pd.DataFrame):
                raise TypeError('ea_data[4] must be pd.DataFrame')
        
    def get_ea_data(self):
        with self.open(mode='rb') as handle:
            content = handle.read()
        return pickle.loads(content)
    
    @property
    def ea_data(self):
        return self.get_ea_data()