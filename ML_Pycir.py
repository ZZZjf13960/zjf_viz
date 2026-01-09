from pycircos import Gcircle, Garc
import matplotlib.pyplot as plt
import collections
import numpy as np

circle = Gcircle(figsize=(8, 8))

networks = [
    'Vis','SomMot','DorsAttn',
    'SalVentAttn','Limbic','Cont','Default'
]

NetworksIdx = {
    'Vis': (1, 17),
    'SomMot': (18, 31),
    'DorsAttn': (32, 46),
    'SalVentAttn': (47, 58),
    'Limbic': (59, 63),
    'Cont': (64, 76),
    'Default': (77, 100)
}

for net in networks:
    start, end = NetworksIdx[net]
    size = end - start + 1

    arc = Garc(
        arc_id=net,
        size=size,
        interspace=3,
        raxis_range=(900, 950),
        label_visible=True,
        
    )
    circle.add_garc(arc)

circle.set_garcs()



