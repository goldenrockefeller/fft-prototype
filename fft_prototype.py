import numpy as np
from scipy.fft import rfft, irfft, fft, ifft

# Real fft by http://www.robinscheibler.org/2013/02/13/real-fft.html

def make_complex_array(x, y):
    return np.array(list(complex(r, i) for r, i in zip(x, y)))

def flatten_complex_signal(complex_signal):
    N = len(complex_signal)
    signal = np.zeros(N * 2)

    for i in range(N):
        signal[2*i] = complex_signal[i].real
        signal[2*i + 1] = complex_signal[i].imag

    return signal

def swap_axes(x):
    return np.conjugate(x) * 1j

def random_complex(size):
    real = np.random.random(size)
    imaj = np.random.random(size)
    comp = make_complex_array(real, imaj)
    return comp

def complement_spectra(spectra):
    conjugate = np.conjugate(spectra)
    complement = conjugate[::-1]

    current_value = complement[-1]
    for i in range(len(complement)):
        next_value = complement[i]
        complement[i] = current_value
        current_value = next_value

    return complement

def my_rfft(signal):
    N = len(signal)
    even_signal = signal[::2]
    odd_signal = signal[1::2]
    combined_signal =  make_complex_array(even_signal, odd_signal)
    combined_spectra = fft(combined_signal)
    the_complement_spectra = complement_spectra(combined_spectra)
    rotor = np.exp(np.arange(0, N//2, dtype=complex) * np.pi * 2 / N * -1j)
    even_spectra = 0.5 * (combined_spectra + the_complement_spectra)
    odd_spectra = -0.5j * (combined_spectra - the_complement_spectra)
    spectra = np.zeros(N//2 + 1, dtype = complex)
    spectra[:N//2] = even_spectra + odd_spectra*rotor
    spectra[N//2] = (even_spectra[0].real - odd_spectra[0].real)
    return spectra

def my_irfft(spectra):
    N = (len(spectra)-1) * 2
    compact_spectra = np.zeros(N//2, dtype=complex)
    compact_spectra[:N//2] = spectra[:N//2]
    compact_spectra[0] += 1j * spectra[N//2]
    the_complement_spectra = complement_spectra(compact_spectra)
    rotor = np.exp(np.arange(0, N//2, dtype=complex) * np.pi * 2 / N * 1j)
    even_spectra = 0.5 * (compact_spectra + the_complement_spectra)
    odd_spectra = 0.5 * ( compact_spectra - the_complement_spectra) * rotor
    ampl_at_zero_freq = even_spectra[0]
    ampl_at_nyquist_freq = -1j * odd_spectra[0]
    even_spectra[0] = 0.5 * (ampl_at_zero_freq + ampl_at_nyquist_freq)
    odd_spectra[0] = 0.5 * (ampl_at_zero_freq - ampl_at_nyquist_freq)
    # Notice the even_spectra[0], even_spectra[N//4] ,
    # odd_spectra[0], odd_spectra[N//4] are all real numbers
    combined_spectra = even_spectra + 1j * odd_spectra
    combined_signal = ifft(combined_spectra)
    signal = flatten_complex_signal(combined_signal)
    return signal

def scrambled_indexes(n_indexes):
    the_scrambled_indexes = [0, n_indexes//2]
    index_adder = n_indexes//2
    n_scrambled_indexes_steps = int(np.log2(n_indexes))

    for i in range(n_scrambled_indexes_steps-1):
        index_adder = index_adder // 2
        next_scrambled_indexes = the_scrambled_indexes.copy()
        for index in the_scrambled_indexes:
            next_scrambled_indexes.append(index + index_adder)
        the_scrambled_indexes = next_scrambled_indexes

    return the_scrambled_indexes

def br_scrambled_indexes(n_indexes):
    the_scrambled_indexes = list(range(0, n_indexes))

    n_bits = int(np.log2(n_indexes))

    for i in range(n_indexes):
        work = i
        br_index = 0
        for bit_id in range(n_bits):
            br_index = br_index << 1;
            br_index += work & 1
            work = work >> 1;
        the_scrambled_indexes[i] = br_index

    return the_scrambled_indexes

def scrambled_signal(scrambled_indexes, signal, scale_factor):
    new_signal = signal.copy()

    for new_index, old_index in enumerate(scrambled_indexes):
        new_signal[new_index] = scale_factor * signal[old_index]

    return new_signal

def br_permute(signal):
    the_scrambled_indexes = br_scrambled_indexes(len(signal))
    return scrambled_signal(the_scrambled_indexes, signal, 1.)

def test1(a, b, n):
    rotor1 = np.exp(np.arange(0, n, dtype=complex)* 2* np.pi / n * -1j)
    rotor2 = np.exp(np.arange(0, n, dtype=complex)* 2* np.pi / n * 1j)
    return a*rotor1 + b*rotor2

def test2(a, b, n):
    total = a+b
    diff = a-b

    rotor = np.exp(np.arange(0, n, dtype=complex)* 2* np.pi / n * -1j)
    real = np.real(rotor)
    imag = np.imag(rotor)

    return total * real + diff * imag * 1j

def dft(signal):
    signal_len = len(signal)
    spectra = 0. * signal.copy()

    for i in range(signal_len):
        spectra += (
            signal[i]
            * myexpe(np.arange(0, signal_len, dtype=float) * 2 * i / signal_len)
        )

    return spectra

def rem2(x):
    return x - np.floor(x/2) * 2

def quarter_sin(x):
    x = rem2(x)

    x = (x <= 0.5) * x + (0.5 < x) * (x <= 1.5) * (1- x) - (x > 1.5) * (2 - x)
    return x;

def quarter_cos(x):
    x = rem2(x)

    x = (x <= 1) * x + (x > 1) * (2 - x)
    return x;


def myexpq(x):
    return np.cos(np.pi * quarter_cos(x)) - np.sin(np.pi * quarter_sin(x)) * 1j

def esin(x):
    return -ecos(x+0.5)

def ecos(x):
    x = rem2(x)

    y = 0. * x

    y += (x < 0.25) * np.cos(np.pi * x)
    y += (x >= 0.25) * (x < 0.75) * np.sin(np.pi * (0.5 - x))
    y += (x >= 0.75) * (x < 1.25) * - np.cos(np.pi * (1-x))
    y += (x >= 1.25) * (x < 1.75) * np.sin(np.pi * (x-1.5))
    y += (x >= 1.75) * np.cos(np.pi * (2-x))

    return y

def myexpr(x):
    return np.cos(np.pi * rem2(x)) - np.sin(np.pi * rem2(x)) * 1j

def myexpx(x):
    return np.cos(np.pi * x) - np.sin(np.pi * x) * 1j

def myexpe(x):
    return ecos(x) - esin(x) * 1j

myexp = myexpe

class Myfft:
    def __init__(self, signal_len, dft_len = 1):
        self.n_radix_4_butterflies = int(np.log2(signal_len/dft_len)) // 2
        self.n_radix_2_butterflies = int(np.log2(signal_len/dft_len))
        self.using_final_radix_2_butterflies = (
            2 * self.n_radix_4_butterflies != self.n_radix_2_butterflies
        )

        self.twiddles = []
        subtwiddle_len = dft_len


        self.dft_len = dft_len

        self.dft_mat = []
        self.dft_scrambled_indexes = scrambled_indexes(dft_len)

        self.dft_mat_t = []


        for dft_basis_id in range(dft_len):
            dft_factor = self.dft_scrambled_indexes[dft_basis_id]
            self.dft_mat.append(
                myexp(np.arange(0, dft_len, dtype=float) * 2 * dft_factor / dft_len),
            )
            self.dft_mat_t.append(
                    myexp(np.array(scrambled_indexes(dft_len)) * 2. * dft_basis_id / dft_len),
            )


        for butterfly_id in range(self.n_radix_4_butterflies):
            self.twiddles.append(
                np.concatenate((
                    myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len),
                    myexp(np.arange(0, subtwiddle_len, dtype=float) / 2 / subtwiddle_len),
                    myexp(np.arange(0, subtwiddle_len, dtype=float) * 1.5 / subtwiddle_len )
                ))
            )
            subtwiddle_len *= 4

        if (self.using_final_radix_2_butterflies):
            self.twiddles.append (
                myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len)
            )

        self.scrambled_indexes = scrambled_indexes(signal_len)
        self.signal_len = signal_len

    def process(self, signal, calculating_inverse = False, bit_reversal = True):
        if calculating_inverse:
            signal = swap_axes(signal)

        scale_factor = 1



        work_signal_a = signal.copy()

        if calculating_inverse:
            scale_factor =  1/self.signal_len
            work_signal_a = scale_factor * work_signal_a

        if bit_reversal:
            work_signal_a = scrambled_signal(self.scrambled_indexes, work_signal_a, 1.)

        work_signal_b = 0. * work_signal_a

        subfft_len = 1
        n_subfft_len = self.signal_len

        #specialty
        if self.dft_len != 1:
            subfft_len *= self.dft_len
            n_subfft_len //= self.dft_len

            a_id = 0
            bb_id = 0
            for subfft_id in range(n_subfft_len):
                for dft_basis_id in range(self.dft_len):
                    # dft_basis_id = 0 specialty, = 1 for all
                    # dft_basis_id = 1 specialty, = real for all
                    b_id = bb_id
                    M = work_signal_a[a_id]
                    dft_basis = self.dft_mat[dft_basis_id]
                    for dft_factor_id in range(self.dft_len):
                        work_signal_b[b_id] += (
                            M
                            *  dft_basis[dft_factor_id]
                        )
                        b_id += 1
                    a_id += 1
                bb_id += self.dft_len


            work_signal_a = work_signal_b
            work_signal_b = work_signal_a.copy()

        twiddle_id = 0

        for butterfly_id in range(self.n_radix_4_butterflies):
            subtwiddle_len = subfft_len
            subfft_len *= 4
            n_subfft_len //= 4

            a_id = 0
            b_id = subtwiddle_len
            c_id = 2 * subtwiddle_len
            d_id = 3 * subtwiddle_len

            for subfft_id in range(n_subfft_len):

                a_id = subfft_id * subfft_len
                b_id = subtwiddle_len + a_id
                c_id = 2 * subtwiddle_len + a_id
                d_id = 3 * subtwiddle_len + a_id

                # Multiply in place
                twiddle_start_id = subfft_id*subfft_len + subtwiddle_len
                end_id = subfft_id*subfft_len + 4 * subtwiddle_len

                # specialty
                if subtwiddle_len!= 1:
                    work_signal_a[twiddle_start_id : end_id] *= self.twiddles[twiddle_id]

                for i in range(subtwiddle_len):
                    # First butterfly.
                    work_signal_b[a_id] = (
                        work_signal_a[a_id]
                        + work_signal_a[b_id]
                    )

                    work_signal_b[b_id] = (
                        work_signal_a[a_id]
                        - work_signal_a[b_id]
                    )
                    work_signal_b[c_id] = (
                        work_signal_a[c_id]
                        + work_signal_a[d_id]
                    )
                    work_signal_b[d_id] = (
                        work_signal_a[c_id]
                        - work_signal_a[d_id]
                    )


                    # Second butterfly.
                    work_signal_a[a_id] = (
                        work_signal_b[a_id]
                        + work_signal_b[c_id]
                    )

                    work_signal_a[b_id] = (
                        work_signal_b[b_id]
                        - 1j * work_signal_b[d_id]
                    )

                    work_signal_a[c_id] = (
                        work_signal_b[a_id]
                        - work_signal_b[c_id]
                    )

                    work_signal_a[d_id] = (
                        work_signal_b[b_id]
                        + 1j * work_signal_b[d_id]
                    )

                    a_id += 1
                    b_id += 1
                    c_id += 1
                    d_id += 1

            twiddle_id += 1

        #end for

        if (self.using_final_radix_2_butterflies):
            subtwiddle_len = subfft_len
            subfft_len *= 2
            n_subfft_len //= 2

            a_id = 0
            b_id = subtwiddle_len


            # Multiply in place
            twiddle_start_id =  subtwiddle_len
            end_id =   2 * subtwiddle_len
            work_signal_a[twiddle_start_id : end_id] *= self.twiddles[twiddle_id]

            for i in range(subtwiddle_len):

                # First butterfly.
                work_signal_b[a_id] = (
                    work_signal_a[a_id]
                    + work_signal_a[b_id]
                )

                work_signal_b[b_id] = (
                    work_signal_a[a_id]
                    - work_signal_a[b_id]
                )

                work_signal_a[a_id] = work_signal_b[a_id]
                work_signal_a[b_id] = work_signal_b[b_id]


                a_id += 1
                b_id += 1


        if calculating_inverse:
            work_signal_a = swap_axes(work_signal_a)

        return work_signal_a

    def process_dif(self, signal, calculating_inverse = False, bit_reversal = True):
        if calculating_inverse:
            signal = swap_axes(signal)

        scale_factor = 1

        if calculating_inverse:
            scale_factor =  1/self.signal_len
            work_signal_a = scale_factor * work_signal_a

        work_signal_a = signal.copy()
        work_signal_b = 0. * work_signal_a

        subfft_len = self.signal_len
        n_subfft_len = 1

        twiddle_id =  len(self.twiddles) - 1

        if (self.using_final_radix_2_butterflies):
            subtwiddle_len = subfft_len // 2

            a_id = 0
            b_id = subtwiddle_len

            for i in range(subtwiddle_len):

                # First butterfly.
                work_signal_b[a_id] = (
                    work_signal_a[a_id]
                    + work_signal_a[b_id]
                )

                work_signal_b[b_id] = (
                    work_signal_a[a_id]
                    - work_signal_a[b_id]
                )

                work_signal_a[a_id] = work_signal_b[a_id]
                work_signal_a[b_id] = work_signal_b[b_id]


                a_id += 1
                b_id += 1

            # Multiply in place
            twiddle_start_id =  subtwiddle_len
            end_id =   2 * subtwiddle_len
            work_signal_a[twiddle_start_id : end_id] *= self.twiddles[twiddle_id]

            subfft_len //= 2
            n_subfft_len *= 2
            twiddle_id -= 1

        for butterfly_id in reversed(range(self.n_radix_4_butterflies)):  #Notice reversed
            subtwiddle_len = subfft_len // 4

            a_id = 0
            b_id = subtwiddle_len
            c_id = 2 * subtwiddle_len
            d_id = 3 * subtwiddle_len

            for subfft_id in range(n_subfft_len):

                a_id = subfft_id * subfft_len
                b_id = subtwiddle_len + a_id
                c_id = 2 * subtwiddle_len + a_id
                d_id = 3 * subtwiddle_len + a_id

                for i in range(subtwiddle_len):

                    # notice switch with work_signal_b and butterfly

                    # Second butterfly.
                    work_signal_b[a_id] = (
                        work_signal_a[a_id]
                        + work_signal_a[c_id]
                    )

                    work_signal_b[b_id] = (
                        work_signal_a[b_id]
                        + work_signal_a[d_id] # notice change
                    )

                    work_signal_b[c_id] = (
                        work_signal_a[a_id]
                        - work_signal_a[c_id]
                    )

                    work_signal_b[d_id] = (
                        -1j * work_signal_a[b_id] # notice change
                        + 1j * work_signal_a[d_id]
                    )


                    # First butterfly.
                    work_signal_a[a_id] = (
                        work_signal_b[a_id]
                        + work_signal_b[b_id]
                    )

                    work_signal_a[b_id] = (
                        work_signal_b[a_id]
                        - work_signal_b[b_id]
                    )
                    work_signal_a[c_id] = (
                        work_signal_b[c_id]
                        + work_signal_b[d_id]
                    )
                    work_signal_a[d_id] = (
                        work_signal_b[c_id]
                        - work_signal_b[d_id]
                    )

                    a_id += 1
                    b_id += 1
                    c_id += 1
                    d_id += 1

                # Multiply in place
                twiddle_start_id = subfft_id*subfft_len + subtwiddle_len
                end_id = subfft_id*subfft_len + 4 * subtwiddle_len

                # specialty
                if subtwiddle_len!= 1:
                    work_signal_a[twiddle_start_id : end_id] *= self.twiddles[twiddle_id]

            twiddle_id -= 1
            subfft_len //= 4
            n_subfft_len *= 4

        #specialty
        if self.dft_len != 1:
            a_id = 0
            bb_id = 0
            for subfft_id in range(n_subfft_len):
                b_id = bb_id
                for dft_factor_id in range(self.dft_len):
                    work_signal_b[b_id] = 0.
                    b_id += 1
                for dft_basis_id in range(self.dft_len):
                    # dft_basis_id = 0 specialty, = 1 for all
                    # dft_basis_id = self.dft_len/2 specialty, = 1 for all
                    b_id = bb_id
                    M = work_signal_a[a_id]
                    dft_basis = self.dft_mat_t[dft_basis_id]
                    for dft_factor_id in range(self.dft_len):
                        work_signal_b[b_id] += (
                            M
                            *  dft_basis[dft_factor_id]
                        )
                        b_id += 1
                    a_id += 1
                bb_id += self.dft_len

            work_signal_a = work_signal_b
            work_signal_b = work_signal_a.copy()

        if bit_reversal:
            work_signal_a = scrambled_signal(self.scrambled_indexes, work_signal_a, 1.)

        if calculating_inverse:
            work_signal_a = swap_axes(work_signal_a)

        return work_signal_a


sig_len =  64

a = random_complex(sig_len)
b = random_complex(sig_len)

# for k in range(0, sig_len, 2):
#     r[k] = (k//2) + ((k//2) & 1)*1.j
#     r[k+1] = -1 - k//2 + ((k//2) & 1)*1.j

myfft = Myfft(sig_len,8)
def mfft(r):
    return myfft.process(r, False)

def mifft(r):
    return myfft.process(r, True)

def mfft_nb(r):
    return myfft.process_dif(r, False, True)

def mifft_nb(r):
    return myfft.process(r, True, True)

f = ifft(fft(a) * fft(b))
m1 = mifft(mfft(a) * mfft(b))
m2 = mifft_nb(mfft_nb(a) * mfft_nb(b))


print(np.max(np.abs(f-m1)), "f-m1")
print(np.max(np.abs(f-m2)), "f-m2")

