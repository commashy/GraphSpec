import math

data = {
  22: 19415,
  26: 9168,
  12: 77355,
  11: 82916,
  10: 82041,
  29: 5391,
  14: 65290,
  13: 71552,
  18: 38737,
  19: 34420,
  56: 10,
  20: 28948,
  16: 53660,
  9: 79577,
  31: 3658,
  49: 73,
  15: 59167,
  30: 4601,
  17: 44608,
  28: 6417,
  8: 69970,
  21: 23830,
  33: 2313,
  36: 1271,
  25: 11316,
  41: 419,
  43: 234,
  7: 51555,
  6: 20237,
  44: 194,
  32: 2888,
  45: 165,
  24: 13495,
  50: 44,
  35: 1454,
  23: 16893,
  27: 7663,
  37: 1150,
  38: 825,
  34: 2089,
  39: 603,
  46: 115,
  40: 627,
  60: 3,
  5: 69,
  42: 298,
  52: 21,
  47: 128,
  53: 16,
  48: 84,
  51: 16,
  55: 4,
  54: 2,
  67: 1,
  58: 2,
  57: 4
}

# Calculate the weighted mean
total_length = sum(length * count for length, count in data.items())
total_count = sum(data.values())
mean = total_length / total_count

# Calculate the weighted variance
variance = sum(count * (length - mean) ** 2 for length, count in data.items()) / total_count

# Calculate the standard deviation
standard_deviation = math.sqrt(variance)

print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", standard_deviation)
