//Numpy array shape [8]
//Min -0.109375000000
//Max 0.109375000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[8];
#else
bias16_t b16[8] = {0.062500, 0.093750, 0.093750, 0.109375, -0.109375, -0.062500, 0.109375, 0.093750};

#endif

#endif
