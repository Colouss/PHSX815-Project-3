import numpy as np
import matplotlib.pyplot as plt
true_limited_p = np.linspace(0.01, 0.99, num=100) #Call the list of true limited probabily to base the program on
#Calling the observered data
rolls_per_slice = 10000
zoomed_prob = 0.2
categorical_prob = 0.02 # Define the probability of the categorical distribution for determining whenever the roll is rolled into the super rare or not
def likelihood(prob, lim, rolls): # The likelihood function used
    return (lim * np.log(prob)) + ((rolls - lim) * np.log(1 - prob))
histogram = np.zeros(shape=(len(true_limited_p)-1, len(true_limited_p)-1)) # Initialize the histogram
for true_p_heads in true_limited_p: # Cycle through true probabilities of heads
    tot_s = 0
    pity = 0
    cate_prob = categorical_prob
    for i in range(rolls_per_slice):  # Iterate through each roll in the categorical part
        if np.random.rand() < cate_prob:  # Check if the roll succeeds
            tot_s = tot_s + 1
            pity = 0  # Reset the counter if the roll succeeds
        else:
            pity += 1  # Increment the counter if the roll fails
            if pity >= 50:  # If the counter reaches 50, add 0.02 to the secondary probability
                cate_prob = 0.02 + categorical_prob   
    measured_limited = np.random.binomial(tot_s, true_p_heads) # Generate the observed data for this slice
    measured_prob = (categorical_prob * true_limited_p) + likelihood(true_limited_p, measured_limited, tot_s) # Generate the measured probability distribution for this slice
    measured_prob -= np.max(measured_prob)  # subtract the maximum value for numerical stability
    measured_prob = np.exp(measured_prob)  # exponentiate to convert back to probabilities
    measured_prob /= np.sum(measured_prob)  # normalize
    likelihood_samples = np.random.choice(true_limited_p, size=10000, p=measured_prob) # Sample from the measured probability distribution
    num_lim_obs = np.random.binomial(tot_s, likelihood_samples) # Generate the number of limited observed to create a confidence interval in a 2D array
    histogram += np.histogram2d(likelihood_samples, num_lim_obs / tot_s,bins=[true_limited_p, true_limited_p])[0] #create the histogram slice, and add it to the histogram

# Initialize the histogram
X, Y = np.meshgrid(true_limited_p, true_limited_p)
plt.imshow(histogram[1:, 1:].T, origin='lower', extent=[true_limited_p[0], true_limited_p[-1], true_limited_p[0], true_limited_p[-1]], 
           aspect='auto', interpolation='bilinear')
plt.xlabel('True Probability of getting a limited')
plt.ylabel('Measured Probability of getting a limited')
plt.colorbar(label='Number of limited rolled that agrees with the probability')
plt.show() #plot the histogram
# Initialize the histogram so that it zooms into a probability (this being at 0.5 as a base)
# Find the index of the specified probability value in the true limited probability array
slice_index = np.argmin(np.abs(true_limited_p - zoomed_prob)) - 1
# Extract the zoomed histogram and corresponding true limited probability values
zoomed_histogram = histogram[slice_index-10:slice_index+11, slice_index-10:slice_index+11]
zoomed_true_limited_p = true_limited_p[slice_index-10:slice_index+11]
zoomed_extent = [zoomed_true_limited_p[0], zoomed_true_limited_p[-1], zoomed_true_limited_p[0], zoomed_true_limited_p[-1]]
plt.imshow(zoomed_histogram.T, origin='lower', extent=zoomed_extent, aspect='auto', interpolation='bilinear')
plt.xlabel('True Probability of getting a limited')
plt.ylabel('Measured Probability of getting a limited')
plt.colorbar(label='Number of limited rolled that agrees with the probability')
plt.title('Zoomed In at the specified probability')
plt.show()
