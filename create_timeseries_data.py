import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import charlie_clean_data as clean_data
import math
from window_slider import Slider

def create_data():
    X_array, Y_array = clean_data.create_arrays()
    final_array =  np.column_stack((X_array, Y_array))

    dates = final_array[:, 0]
    difference = pd.Series(dates)-dates[0]

    df_diff = pd.DataFrame(difference) #find diff since original date
    df_diff['weeks'] = df_diff / pd.Timedelta(weeks=1) # convert to weeks
    weeks_array = pd.DataFrame(df_diff['weeks'])
    weeks_array = np.array(weeks_array.values.tolist()) # convert back to numpy


    final_array = np.delete(final_array, 0, axis=1) # remove the original datetime column

    # Concatenate along the second axis (i.e. columns)
    final_array = np.concatenate((weeks_array, final_array), axis=1)
    final_array = final_array[final_array[:, 0].argsort()]
    
    print(final_array)
    return final_array

def create_windows():
    trainset = create_data().T
   # bucket_size = 5
   # overlap_count = 1
   # slider = Slider(bucket_size,overlap_count)
  #  slider.fit(trainset)       
   # while True:
    #    window_data = slider.slide()
        # do your stuff
     #   print(window_data)
      #  if slider.reached_end_of_list(): break

"""w = 5
train_constructor = WindowSlider()
train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

test_constructor = WindowSlider()
test_windows = test_constructor.collect_windows(testset.iloc[:,1:],
                                                previous_y=False)

train_constructor_y_inc = WindowSlider()
train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=True)

test_constructor_y_inc = WindowSlider()
test_windows_y_inc = test_constructor_y_inc.collect_windows(testset.iloc[:,1:],
                                                previous_y=True)

train_windows.head(3)"""

create_windows()