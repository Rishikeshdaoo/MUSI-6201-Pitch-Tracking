# MUSI-6201-Assignment_1

## Discussions
### B.1.4
One possible reason is insufficient length of block size that is required to resolve resolve f0. Say for instance, the block and hop size are both 441. In this case, the f0 will jump between 441Hz and 445Hz for a ground truth of 441Hz. This is because, the distance between 2 peaks is fluctuating between 100 and 99. If it is 100, the predicted f0 will be 441Hz and if it is 99, the predicted f0 will be ~445.45Hz. The deviation is high with the resolution is low (small block size). As the block size increases, the peaks detected are closer to the real peaks and the the fluctuation -> 0.

One other possible reason is when the block size is too small. In this case, the peaks are not found within the block due to lack of enough information.

### B.4
The errCentRms is 643.862 which is in the expected expected range of 500 - 700. If we plot the predictions and annotations, we can see clear spikes in the prediction. This could be one reason for the large error. This error value is pretty high for practical use.

## Ideas behind the modification of Pitch Tracker:

### Interpolation to ACF curve
The calculation of fundamental frequency is based on the locations of ACF peaks. To determine the peaks of ACF more accurately, instead of directly comparing neighboring values, I do one more step: apply parabolic interpolation around the maximum of the ACF curve. In particular, if n0 is the maximum index of an ACF curve, then we can fit a parabola that passes three points (n0−1,ACF(n0−1)), (n0,ACF(n0)), and (n0+1,ACF(n0+1)), and then use the maxmizing position of this parabola as the real peak to compute the pitch.<br><br>
This idea is implemented via the function "get_f0_from_acfmod" and "parabolic". In the function "get_f0_from_acfmod", first the ACF are sorted to find the maximum value and corresponding index. Then the function "parabolic" is called to do parabolic interpolation around the maximum of the ACF curve. The result of parabolic interpolation is then used as the real maximum position to compute the f0.

### Medium filter
To smooth the pitch curve such that abrupt-changing pitches are removed. Set the kernel size to 7 or 9.<br><br>
This idea is implemented via the function "track_pitch_acfmod". Median filtering is done before outputing the pitch.

### Hamming window in audio blocking
Hamming window is also used to reduce the abrupt-changing pitches. This idea is implemented via the function "block_audio_mod". 

### Others
Except the functions mentioned above, namely "get_f0_from_acfmod", "parabolic", "track_pitch_acfmod" and "run_evaluation_mod", other functions used in pitch tracker evaluation are not modified.

## Result comparison of the provided train data (3 audio clips)
The modified code significantly reduces irrelevant pulses in pitch tracking, making the pitch tracker curve (blue) closer to the ground truth (orange). <br><br>
Result for the first audio clip:<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/1_1.png)<br>
Result for the first audio clip: (after modification)<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/1_2.png)<br>
Result for the second audio clip:<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/2_1.png)<br>
Result for the second audio clip: (after modification)<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/2_2.png)<br>
Result for the third audio clip:<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/3_1.png)<br>
Result for the third audio clip: (after modification)<br>
![](https://github.com/Aavu/MUSI-6201-Assignment_1/blob/Charles/3_2.png)<br>
