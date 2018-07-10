
# Capsule Network - ver.PyTorch

# Prerequisites

- [Python]
- [PyTorch]
- [Visdom](https://github.com/facebookresearch/visdom)
- [NumPy](http://www.numpy.org/)

# Dataset

You have to construct suitable dataloader for your dataset. For example, we generate our dataloader for below data structure.

data

Classes = ['c', 'g', 'h']

    |---Train
        |---c  
		
        |---g
		
        |---h
		

        
    |---Validation
        |---c      
		
        |---g
		
        |---h
		

		
    |---Test
        |---c        
		
        |---g
		
        |---h
		




# Visualizatoin

-  you should run visdom before training
```bash
python -m visdom.server
```
click the URL http://localhost:8097

    
# Acknowledgments

This work was supported by Kakao Corp. and Kakao Brain Corp.