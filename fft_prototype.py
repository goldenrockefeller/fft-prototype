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

def scrambled_signal(scrambled_indexes, signal):
    new_signal = signal.copy()

    for new_index, old_index in enumerate(scrambled_indexes):
        new_signal[new_index] = signal[old_index]

    return new_signal

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
            * np.exp(np.arange(0, signal_len, dtype=complex) * 2 * np.pi * i / signal_len * -1j)
        )

    return spectra


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

        for dft_basis_id in range(dft_len):
            dft_factor = self.dft_scrambled_indexes[dft_basis_id]
            self.dft_mat.append(
                np.exp(np.arange(0, dft_len, dtype=complex) * 2 * np.pi * dft_factor / dft_len * -1j),
            )


        for butterfly_id in range(self.n_radix_4_butterflies):
            self.twiddles.append(
                np.concatenate((
                    np.exp(np.arange(0, subtwiddle_len, dtype=complex) * np.pi / subtwiddle_len * -1j),
                    np.exp(np.arange(0, subtwiddle_len, dtype=complex) * np.pi / 2 / subtwiddle_len * -1j),
                    np.exp(np.arange(0, subtwiddle_len, dtype=complex) * 1.5 * np.pi  / subtwiddle_len * -1j)
                ))
            )
            subtwiddle_len *= 4

        if (self.using_final_radix_2_butterflies):
            self.twiddles.append (
                np.exp(np.arange(0, subtwiddle_len, dtype=complex) * np.pi / subtwiddle_len * -1j)
            )


        self.scrambled_indexes = scrambled_indexes(signal_len)
        self.signal_len = signal_len

    def process(self, signal, calculating_inverse = False):
        if calculating_inverse:
            signal = np.conjugate(signal) / len(signal)

        work_signal_a = scrambled_signal(self.scrambled_indexes, signal)
        work_signal_b = 0. * work_signal_a

        subfft_len = 1
        n_subfft_len = self.signal_len
        twiddle_id = 0

        if self.dft_len != 1:

            subfft_len *= self.dft_len
            n_subfft_len //= self.dft_len

            dft_start_id = 0
            for subfft_id in range(n_subfft_len):
                a_id = dft_start_id
                for dft_basis_id in range(self.dft_len):
                    b_id = dft_start_id
                    for dft_factor_id in range(self.dft_len):
                        work_signal_b[b_id] += (
                            work_signal_a[a_id]
                            *  self.dft_mat[dft_basis_id][dft_factor_id]
                        )
                        b_id += 1
                    a_id += 1

                dft_start_id += subfft_len

            work_signal_a = work_signal_b
            work_signal_b = work_signal_a.copy()

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

            # for subfft_id in range(n_subfft_len): #range(1) for last butterfly
            #     # Multiply in place
            #     twiddle_start_id =  subfft_id*subfft_len + subtwiddle_len
            #     end_id =   subfft_id*subfft_len + 2 * subtwiddle_len
            #     work_signal_a[twiddle_start_id : end_id] *= self.twiddles[twiddle_id]
            #
            #     for i in range(subtwiddle_len):
            #
            #         # First butterfly.
            #         work_signal_b[a_id] = (
            #             work_signal_a[a_id]
            #             + work_signal_a[b_id]
            #         )
            #
            #         work_signal_b[b_id] = (
            #             work_signal_a[a_id]
            #             - work_signal_a[b_id]
            #         )
            #
            #         work_signal_a[a_id] = work_signal_b[a_id]
            #         work_signal_a[b_id] = work_signal_b[b_id]
            #
            #
            #         a_id += 1
            #         b_id += 1
            #
            #     a_id += subfft_len - subtwiddle_len
            #     b_id += subfft_len - subtwiddle_len
            #
            # twiddle_id += 1

        if calculating_inverse:
            work_signal_a = np.conjugate(work_signal_a)

        return work_signal_a


sig_len =  2 ** 15
r = random_complex(sig_len)
f = fft(r)
myfft = Myfft(sig_len, 4)
m = myfft.process(r)
print(np.average(np.abs(f-m)), "f-m")