# Video Labelling Tool Using youtube 8M dataset

## Preparation

Python 2.x was tested to work well.

Prepare necessary libraries for image processing and deep learning.

```
conda install -c conda-forge imageio=2.2.0

pip install -r requirements.txt
```


## Predict a Video Label

1. Prepare a movie data
    - Example: Download a movie data from youtube 
        ```
        pytube https://www.youtube.com/watch?v=korUVQbEA98&t=11s -f YOUR_MOVIE_FILENAME
        ```

2. Predict a label of the movie
```
python video_label_prediction.py YOUR_MOVIE_FILENAME_WITH_EXTENSION  
```


## References
- Keras Inception model
    - https://github.com/fchollet/deep-learning-models

- Youtube 8M starter code
    - https://github.com/google/youtube-8m

    
   