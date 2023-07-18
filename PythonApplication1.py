import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def read_data():
    values = []
    with open("Data.txt", 'r') as file:
        data = file.read()
        values = data.split('\n')
        # Delete the last element, which is an empty string.
        values = values[:-1]   

    floatValues = np.array([float(num) for num in values]) # Strings to floats typecast put in a np array
    return floatValues

def average_rectify(data): 

    # Get absolute values of data.
    absData = np.array([])
    for x in data:
        absData = np.append(absData, abs(float(x)))

    # Apply moving average filter of window size T = 20.
    T = 20
    h = [1/T] * T
    averagedData = np.convolve(absData, h, 'same')
    return averagedData

def detect_MUAPs(data):
    dataAverage = average_rectify(data)
    threshold = 11.7
    T = 20
    i = 0
    startIndices = []
    endIndices   = []

    # Detecting MUAPs
    while i < len(dataAverage):
        if dataAverage[i] > threshold:
            potentialStart = i
            for j in range(0, T):
                if dataAverage[i+j] < threshold:
                    potentialStart = -1
                    break
            if potentialStart != -1:
                startIndices.append(potentialStart)
                while dataAverage[i] > threshold:
                    i += T+1
                endIndices.append(i)
        i += 1

    # Align peaks to the middle of the MUAP.
    peakIndex = []
    for i in range(len(startIndices)):
        peakIndex.append(np.argmax(data[startIndices[i]:endIndices[i]]) + startIndices[i])

    for i in range(len(startIndices)):
        startIndices[i] = peakIndex[i] - 10
        endIndices[i]   = peakIndex[i] + 10
    return startIndices, endIndices, peakIndex

def splice_detected_MUAPs(startIndices, endIndices, data):
    #detectedMUAPsWhole is the original signal but only our detected MUAPs are shown on it for comparison sake
    #detectedMUAPs      is an array containing our MUAPs as numpy arrays for processing
    detectedMUAPsWhole  = np.array([0.0]*len(data)) # Initializing zero signal
    detectedMUAPs       = []          # Initializing array of numpy arrays

    for i in range(len(startIndices)):            
        detectedMUAPsWhole[startIndices[i]:endIndices[i]] = data[startIndices[i]:endIndices[i]] 
        detectedMUAPs.append(data[startIndices[i]:endIndices[i]]) # Appending our detected MUAPs as numpy arrays

    return detectedMUAPsWhole, detectedMUAPs
        
def sum_of_difference_square(template, muap):
    result = 0
    for i in range(len(template)):
        result += pow(template[i] - muap[i], 2)
    return result

# Updating template by averaging with matched MUAP.
def template_average(template, muap):
    averaged = []
    for i in range(len(template)):
        averaged.append( (template[i] + muap[i]) / 2 )
    return averaged

# Template generator defines two Difference Thresholds
# The higher threshold is for high amplitude MUAPs
# returns templates and an array MUAPsClassification that contains each muap calssification
# MUAPsClassification linked with muaps array by index
def template_generator(muaps):
    HighDifferenceThreshold = pow(14.65, 5)
    LowDifferenceThreshold = pow(9.35, 5) 

    MUAPsClassifications = []
    templates = [np.array([])]
    templates[0] = muaps[0]

    for i in range(0, len(muaps)):
        found = 0
        t = 0
        savedThresholds = [HighDifferenceThreshold]*len(templates)

        # If muap matches a template we update the template by computing its average with the matched muap
        #   and update the MUAPsClassification
        # If not we make a new template and a new classification denoted by a new integer which is the the index the
        #   new template
        for j in range(0, len(templates)):
            if np.max(muaps[i]) > 200:
                if sum_of_difference_square(templates[j], muaps[i]) < HighDifferenceThreshold:
                    savedThresholds[j] = sum_of_difference_square(templates[j], muaps[i])
                    found = 1
            else:
                if sum_of_difference_square(templates[j], muaps[i]) < LowDifferenceThreshold:
                    savedThresholds[j] = sum_of_difference_square(templates[j], muaps[i])
                    found = 1

        if found == 1:
            index = np.argmin(savedThresholds)
            template = template_average(templates[index], muaps[i])
            MUAPsClassifications.append(index)
        else:
            templates.append(muaps[i])
            MUAPsClassifications.append(len(templates)-1)
       
    return templates, MUAPsClassifications     


def plot_marked_MUAPs(data, startIndices, MUAPsClassifications):
    # Plot markers with colors based on values
    MarkFigure = plt.figure()
    plt.plot(data[30000:35000])
    splicedIndices = []             # list containing the indices of in startIndices that are within the sample range
    splicedMUAPsClassified = []     # list containing the types of MUAPs corresponding to the spliced Indices

    # Initializing both lists
    for i in range(len(startIndices)):
        if startIndices[i] >= 30000 and startIndices[i] <= 35000:
            splicedIndices.append(startIndices[i] - 30000)          # -30000 to align it with wave, since Indices start at sample 30000
            splicedMUAPsClassified.append(MUAPsClassifications[i])

    # Plot them
    plt.scatter(splicedIndices, [np.max(data) + 50]*len(splicedIndices), c=splicedMUAPsClassified, cmap='viridis')
    plt.title("Samples from 30000 to 35000 with MUAPs marked by type")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (microV)")

def plot_templates(templates):
    i = 1
    print("plotting templates ...")
    for t in templates:
        plt.figure(i)
        plt.plot(t)
        plt.title(f"template {i}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (microV)")
        i += 1

        # To avoid crashing computers with matlibplot we have an upper bound to the amount of templates that can be shown
        # in case for some unfortunate reason we detected more than 3 templates
        if i >= 20:
            break

def plot_spectrums(startIndices, peakIndices, MUAPsClassified, data):
    # Getting binary vectors
    i = 0
    binaryVecs = [[0]*len(data)] *len(templates)

    while i < len(data):
        if i in startIndices:
            linkingIndex = startIndices.index(i)                    # linkingIndex is the Index that linkes our three arrays together
            typeOfMUAP = MUAPsClassified[linkingIndex]              # typeOfMUAP determines the vector that is changed in binaryVecs
            binaryVecs[typeOfMUAP] = binaryVecs[typeOfMUAP].copy()  # Create a copy of the row so not all rows change at once
            
            # From detection to end of detection mark the spectrum to be 1
            binaryVecs[typeOfMUAP][peakIndices[linkingIndex]] = 1
            i += 1                                                  # Incrementing i by the range of detection
            continue
        i += 1

    # figure number for matlibplot plots, different number = different window for the plot to appear in
    f = 5
    spectrums = [[]] *len(templates)
    spectrumsFrequencies = [[]] * len(templates)
    # Computing and plotting their dft spectrums
    for i in range(len(binaryVecs)):
        # Fast fourier Transform of spectrums
        spectrums[i] = np.fft.fft(binaryVecs[i])
        # Getting the frequency for x axis
        spectrumsFrequencies[i] = np.fft.fftfreq(len(spectrums[i]))

        plt.figure(f)
        plt.scatter(spectrumsFrequencies[i], np.abs(spectrums[i]), s=4)
        plt.title(f"Spectrum of Frequencies of template {i + 1}")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        f += 1
    
    
data = read_data()

startIndices, endIndices, peakIndices = detect_MUAPs(data)
detectedMUAPsWhole, detectedMUAPs = splice_detected_MUAPs(startIndices, endIndices, data)
print("number of MUAPs: ", len(detectedMUAPs))

templates, MUAPsClassifications = template_generator(detectedMUAPs)
print("number of templates: ", len(templates))

plot_templates(templates)

plot_spectrums(startIndices, peakIndices, MUAPsClassifications, data)

plot_marked_MUAPs(data, startIndices, MUAPsClassifications)

# Print the the frequency of each type appearring in the classification list
numberOfType = Counter(MUAPsClassifications)

for num, count in numberOfType.items():
    print(f"type {num + 1} appears {count} time(s).")


# Display all plots
plt.show()

