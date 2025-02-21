import numpy as np
from enum import Enum, auto
from scipy.fft import rfft, irfft, fft, ifft

# Real fft by http://www.robinscheibler.org/2013/02/13/real-fft.html

#Strategies:
"""
Deinterleave-Interleave
Extend-Multiply-Interleave
Duplicate-Interleave
"""

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

def complement_spectra_2(spectra):
    # spectra len is even
    spectra_len = len(spectra);
    half_spectra_len = (spectra_len) // 2

    complement = spectra * 0.

    complement[0] = spectra[0]
    complement[half_spectra_len] =  spectra[half_spectra_len]

    for i in range(1, half_spectra_len):
        complement[i]  = spectra[spectra_len-i]
        complement[spectra_len-i] = spectra[i]

    return complement

def my_rfft(signal):
    N = len(signal)
    even_signal = signal[::2]
    odd_signal = signal[1::2]
    combined_signal =  make_complex_array(even_signal, odd_signal) # work

    combined_spectra = fft(combined_signal)
    print(combined_spectra)


    the_complement_spectra = complement_spectra(combined_spectra) # work
    print(the_complement_spectra)
    rotor = np.exp(np.arange(0, N//2, dtype=complex) * np.pi * 2 / N * -1j) # save
    spectra = np.zeros(N//2 + 1, dtype = complex)

    spectra[:N//2] = 0.5 * (
        (combined_spectra + the_complement_spectra)
         - 1j *  (combined_spectra - the_complement_spectra) * rotor
    )
    #spectra[:N//2] = even_spectra + odd_spectra*rotor #the even spectra
    spectra[N//2] = (
        (0.5 * (combined_spectra[0].real + the_complement_spectra[0].real))
        - (0.5 * (combined_spectra[0].imag - the_complement_spectra[0].imag))
        )
    return spectra

def my_irfft(spectra):
    N = (len(spectra)-1) * 2
    compact_spectra = np.zeros(N//2, dtype=complex)
    compact_spectra[:N//2] = spectra[:N//2]
    compact_spectra[0] += 1j * spectra[N//2]
    ampl_at_zero_freq = spectra[0]
    ampl_at_nyquist_freq = spectra[N//2]

    print(compact_spectra)

    the_complement_spectra = complement_spectra(compact_spectra)
    rotor = np.exp(np.arange(0, N//2, dtype=complex) * np.pi * 2 / N * 1j)
    even_spectra = 0.5 * (compact_spectra + the_complement_spectra)
    odd_spectra = 0.5 * ( compact_spectra - the_complement_spectra) * rotor
    even_spectra[0] = 0.5 * (ampl_at_zero_freq + ampl_at_nyquist_freq)
    odd_spectra[0] = 0.5 * (ampl_at_zero_freq - ampl_at_nyquist_freq)
    # Notice the even_spectra[0], even_spectra[N//4] ,
    # odd_spectra[0], odd_spectra[N//4] are all real numbers
    combined_spectra = even_spectra + 1j * odd_spectra

    compact_spectra[0] =  (
        0.5 * (ampl_at_zero_freq + ampl_at_nyquist_freq)
        + 0.5j * (ampl_at_zero_freq - ampl_at_nyquist_freq)
    )

    print(combined_spectra)
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

def scrambled_flip(signal):
    start_id = 2
    section_len = 2
    new_signal = [0] * len(signal)
    new_signal[0] = signal[0]
    new_signal[1] = signal[1]
    while start_id < len(signal):
        a = start_id
        b = start_id + section_len - 1
        for i in range(section_len):
            new_signal[a] = signal[b]
            a += 1
            b -= 1
        start_id += section_len
        section_len = section_len << 1
    return new_signal



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
#
# bit_reversed_indexes = scrambled_indexes(64)
# for i in range(4):
#     print(
#
#         f"q0_real = Operand::load_unaligned(in_real + {4*i});",
#         f"q1_real = Operand::load_unaligned(in_real + {4*i+16});",
#         f"q2_real = Operand::load_unaligned(in_real + {4*i+32});",
#         f"q3_real = Operand::load_unaligned(in_real + {4*i+18});",
#
#         f"q0_imag = Operand::load_unaligned(in_imag + {4*i});",
#         f"q1_imag = Operand::load_unaligned(in_imag + {4*i+16});",
#         f"q2_imag = Operand::load_unaligned(in_imag + {4*i+32});",
#         f"q3_imag = Operand::load_unaligned(in_imag + {4*i+18});",
#
#         f"p0_real = xsimd::zip_lo(q0_real, q1_real);",
#         f"p1_real = xsimd::zip_hi(q0_real, q1_real);",
#         f"p2_real = xsimd::zip_lo(q2_real, q3_real);",
#         f"p3_real = xsimd::zip_hi(q2_real, q3_real);",
#
#         f"p0_imag = xsimd::zip_lo(q0_imag, q1_imag);",
#         f"p1_imag = xsimd::zip_hi(q0_imag, q1_imag);",
#         f"p2_imag = xsimd::zip_lo(q2_imag, q3_imag);",
#         f"p3_imag = xsimd::zip_hi(q2_imag, q3_imag);",
#
#         f"r0_real = xsimd::zip_lo(p0_real, p2_real);",
#         f"r1_real = xsimd::zip_hi(p0_real, p2_real);",
#         f"r2_real = xsimd::zip_lo(p1_real, p3_real);",
#         f"r3_real = xsimd::zip_hi(p1_real, p3_real);",
#
#         f"r0_imag = xsimd::zip_lo(p0_imag, p2_imag);",
#         f"r1_imag = xsimd::zip_hi(p0_imag, p2_imag);",
#         f"r2_imag = xsimd::zip_lo(p1_imag, p3_imag);",
#         f"r3_imag = xsimd::zip_hi(p1_imag, p3_imag);",
#
#         f"r0_real.store_unaligned(out_real + {bit_reversed_indexes[4*i]});",
#         f"r1_real.store_unaligned(out_real + {bit_reversed_indexes[4*i + 1]});",
#         f"r2_real.store_unaligned(out_real + {bit_reversed_indexes[4*i + 2]});",
#         f"r3_real.store_unaligned(out_real + {bit_reversed_indexes[4*i + 3]});",
#
#         f"r0_imag.store_unaligned(out_imag + {bit_reversed_indexes[4*i]});",
#         f"r1_imag.store_unaligned(out_imag + {bit_reversed_indexes[4*i + 1]});",
#         f"r2_imag.store_unaligned(out_imag + {bit_reversed_indexes[4*i + 2]});",
#         f"r3_imag.store_unaligned(out_imag + {bit_reversed_indexes[4*i + 3]});",
#         sep="\n"
#     )

# Transposing Bitreversal:
# [ ] How does it compare in speed to COBRA?
# [-] Construct indexes as a matrix (or two if matrix is rectangular
# [-] Bit reverse permute the rows
# [-] Recursive transpose to spit out load and store orders
# [ ] Use a counter to determine how many times off-diagonal transposes with happen
# [ ] Have a templated method for transposition.

def get_indexes_as_mats(n_indexes):
    n_bits = int(np.log2(n_indexes))

    if n_bits % 2 == 0:
        # This is a square matrix
        n_rows = 1 << (n_bits // 2)
        mat = []
        for row_id in range(n_rows):
            row = [0] * n_rows
            for col_id in range(n_rows):
                row[col_id] = row_id * n_rows + col_id
            mat.append(row)
        return [mat]

    else:
        # This is a rectangular matrix
        n_rows = 1 << ((n_bits - 1) // 2)
        mat_a = []
        mat_b = []
        for row_id in range(n_rows):
            row_a = [0] * n_rows
            row_b = [0] * n_rows
            for col_id in range(n_rows):
                row_a[col_id] = row_id * 2 * n_rows + col_id
                row_b[col_id] = row_id * 2 * n_rows + col_id + n_rows
            mat_a.append(row_a)
            mat_b.append(row_b)
        return [mat_a, mat_b]

def bit_rev_permute_rows(mat):
    n_rows = len(mat)
    new_mat = []

    scrambled_rows = scrambled_indexes(n_rows)

    for row_id in range(n_rows):
        new_mat.append(mat[scrambled_rows[row_id]])

    return new_mat

def corner(mat, quarter_row, quater_col):

    new_n_rows = len(mat) // 2

    new_mat = []

    for row_id in range(new_n_rows):
        old_row = mat[new_n_rows * quarter_row + row_id]
        new_row = []
        for col_id in range(new_n_rows):
            new_row.append(old_row[new_n_rows * quater_col + col_id])
        new_mat.append(new_row)

    return new_mat

def transpose_diagonal_indexes(mat, base_size):
    if len(mat) == base_size:
        destinations = []
        for row_id in range(base_size):
            destinations.append(mat[row_id][0])
        return [[destinations, destinations]]

    destinations = []
    destinations.extend(transpose_diagonal_indexes(corner(mat,0,0), base_size))
    destinations.extend(
        transpose_off_diagonal_indexes(
            corner(mat,0,1),
            corner(mat,1,0),
            base_size
        )
    )
    destinations.extend(transpose_diagonal_indexes(corner(mat,1,1), base_size))
    return destinations

def transpose_off_diagonal_indexes(mat_a, mat_b, base_size):
    if len(mat_a) == base_size:
        destinations_a = []
        destinations_b = []
        for row_id in range(base_size):
            destinations_a.append(mat_a[row_id][0])
            destinations_b.append(mat_b[row_id][0])
        return [[destinations_a, destinations_b]]

    destinations = []
    destinations.extend(
        transpose_off_diagonal_indexes(
            corner(mat_a,0,0),
            corner(mat_b,0,0),
            base_size
        )
    )
    destinations.extend(
        transpose_off_diagonal_indexes(
            corner(mat_a,0,1),
            corner(mat_b,1,0),
            base_size
        )
    )
    destinations.extend(
        transpose_off_diagonal_indexes(
            corner(mat_a,1,0),
            corner(mat_b,0,1),
            base_size
        )
    )
    destinations.extend(
        transpose_off_diagonal_indexes(
            corner(mat_a,1,1),
            corner(mat_b,1,1),
            base_size
        )
    )
    return destinations

def get_off_diagonal_streaks(transpose_pairs):
    streaks = []
    counter = 0
    for pair in transpose_pairs:
        if pair[0][0] == pair[1][0]:
            if counter != 0:
                if counter != 1:
                    streaks.append(counter)
                counter = 0
        else:
            counter += 1

    return streaks

def print_mat(mat):
    print(np.array(mat))

class PlanType(Enum):
    N_INDEXES_BASE_SIZE_SQR = auto()
    N_INDEXES_2_BASE_SIZE_SQR = auto()
    N_INDEXES_4_BASE_SIZE_SQR = auto()
    N_INDEXES_8_BASE_SIZE_SQR = auto()
    MAT_IS_SQUARE = auto()
    MAT_IS_NONSQUARE = auto()


class BitRevPermPlan:
    type : PlanType
    plan_indexes : list[int]
    off_diagonal_streak_lens = list[int]


def get_bit_rev_perm_plan(n_indexes, base_size):
    n_bits = int(np.log2(n_indexes))

    mat_is_square = n_bits % 2 == 0

    plan = BitRevPermPlan()

    if n_indexes ==  base_size * base_size:
        plan.type = PlanType.N_INDEXES_BASE_SIZE_SQR

    elif n_indexes ==  2* base_size * base_size:
        plan.type = PlanType.N_INDEXES_2_BASE_SIZE_SQR

    elif n_indexes ==  4* base_size * base_size:
        plan.type = PlanType.N_INDEXES_4_BASE_SIZE_SQR

    elif n_indexes ==  8* base_size * base_size:
        plan.type = PlanType.N_INDEXES_8_BASE_SIZE_SQR


    elif mat_is_square:
        plan.type = PlanType.MAT_IS_SQUARE

    else:
        plan.type = PlanType.MAT_IS_NONSQUARE

    mats = get_indexes_as_mats(n_indexes)



    pre_plans_indexes = []

    for mat in mats:
        mat = bit_rev_permute_rows(mat)
        pre_plans_indexes.append(transpose_diagonal_indexes(mat, base_size))

    off_diagonal_streak_lens = get_off_diagonal_streaks(pre_plans_indexes[0])

    plan_indexes = []

    # Compress diagonal

    pair_id = 0

    for pre_plan in pre_plans_indexes:
        for j in range(base_size):
            plan_indexes.append(pre_plan[pair_id][0][j])

    pair_id += 1

    if n_indexes ==  base_size * base_size or n_indexes ==  2* base_size * base_size:
        plan.plan_indexes = plan_indexes
        plan.off_diagonal_streak_lens = []
        return plan

    for pre_plan in pre_plans_indexes:
        for j in range(base_size):
            plan_indexes.append(pre_plan[pair_id][0][j])
        for j in range(base_size):
            plan_indexes.append(pre_plan[pair_id][1][j])

    pair_id += 1

    for pre_plan in pre_plans_indexes:
        for j in range(base_size):
            plan_indexes.append(pre_plan[pair_id][0][j])

    pair_id += 1

    for streak_len in off_diagonal_streak_lens:
        for off_diagonal_id in range(streak_len):
            for pre_plan in pre_plans_indexes:
                for j in range(base_size):
                    plan_indexes.append(pre_plan[pair_id][0][j])
                for j in range(base_size):
                    plan_indexes.append(pre_plan[pair_id][1][j])
            pair_id += 1

        for pre_plan in pre_plans_indexes:
            for j in range(base_size):
                plan_indexes.append(pre_plan[pair_id][0][j])

        pair_id += 1

        for pre_plan in pre_plans_indexes:
            for j in range(base_size):
                plan_indexes.append(pre_plan[pair_id][0][j])
            for j in range(base_size):
                plan_indexes.append(pre_plan[pair_id][1][j])

        pair_id += 1

        for pre_plan in pre_plans_indexes:
            for j in range(base_size):
                plan_indexes.append(pre_plan[pair_id][0][j])

        pair_id += 1

    plan.plan_indexes = plan_indexes
    plan.off_diagonal_streak_lens = off_diagonal_streak_lens
    return plan


def data_transpose_diagonal(data, indexes, base_size):
    for i in range(base_size):
        for j in range(i + 1, base_size):
            data[indexes[i] + j], data[indexes[j] + i] = (
                data[indexes[j] + i], data[indexes[i] + j]
            )

def data_transpose_off_diagonal(data, indexes0, indexes1, base_size):
    for i in range(base_size):
        for j in range(base_size):
            data[indexes0[i] + j], data[indexes1[j] + i] = (
                data[indexes1[j] + i], data[indexes0[i] + j]
            )


def transpose_scrambled_indexes(n_indexes, base_size):
    indexes = list(range(n_indexes))

    plan = get_bitrevperm_plan(n_indexes, base_size)
    plan_type, plan_indexes, off_diagonal_streak_lens  = plan.plan_indexes, plan.off_diagonal_streak_lens

    transpose_id = 0

    if plan_type is PlanType.N_INDEXES_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes, base_size)
        return indexes

    if plan_type is PlanType.N_INDEXES_2_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes[0:base_size], base_size)
        data_transpose_diagonal(indexes, plan_indexes[base_size:], base_size)
        return indexes

    if plan_type is PlanType.N_INDEXES_4_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes[0:base_size], base_size)
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[base_size:2*base_size],
            plan_indexes[2*base_size:3*base_size],
            base_size
        )
        data_transpose_diagonal(indexes, plan_indexes[3*base_size:], base_size)
        return indexes

    if plan_type is PlanType.N_INDEXES_8_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes[0:base_size], base_size)
        data_transpose_diagonal(indexes, plan_indexes[base_size:2*base_size], base_size)
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[2*base_size:3*base_size],
            plan_indexes[3*base_size:4*base_size],
            base_size
        )
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[4*base_size:5*base_size],
            plan_indexes[5*base_size:6*base_size],
            base_size
        )
        data_transpose_diagonal(indexes, plan_indexes[6*base_size:7*base_size], base_size)
        data_transpose_diagonal(indexes, plan_indexes[7*base_size:], base_size)
        return indexes

    if plan_type is PlanType.MAT_IS_SQUARE:
        data_transpose_diagonal(indexes, plan_indexes[0:base_size], base_size)
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[base_size:2*base_size],
            plan_indexes[2*base_size:3*base_size],
            base_size
        )
        data_transpose_diagonal(indexes, plan_indexes[3*base_size:4*base_size], base_size)
        transpose_id = 4

        for streak_len in off_diagonal_streak_lens:
            for off_diagonal_id in range(streak_len):
                data_transpose_off_diagonal(
                    indexes,
                    plan_indexes[transpose_id*base_size:(transpose_id+1)*base_size],
                    plan_indexes[(transpose_id+1)*base_size:(transpose_id+2)*base_size],
                    base_size
                )
                transpose_id += 2

            data_transpose_diagonal(indexes, plan_indexes[(transpose_id)*base_size:(transpose_id+1)*base_size], base_size)
            data_transpose_off_diagonal(
                indexes,
                plan_indexes[(transpose_id+1)*base_size:(transpose_id+2)*base_size],
                plan_indexes[(transpose_id+2)*base_size:(transpose_id+3)*base_size],
                base_size
            )
            data_transpose_diagonal(indexes, plan_indexes[(transpose_id+3)*base_size:(transpose_id+4)*base_size], base_size)

            transpose_id += 4

        return indexes

    data_transpose_diagonal(indexes, plan_indexes[0:base_size], base_size)
    data_transpose_diagonal(indexes, plan_indexes[base_size:2*base_size], base_size)
    data_transpose_off_diagonal(
        indexes,
        plan_indexes[2*base_size:3*base_size],
        plan_indexes[3*base_size:4*base_size],
        base_size
    )
    data_transpose_off_diagonal(
        indexes,
        plan_indexes[4*base_size:5*base_size],
        plan_indexes[5*base_size:6*base_size],
        base_size
    )
    data_transpose_diagonal(indexes, plan_indexes[6*base_size:7*base_size], base_size)
    data_transpose_diagonal(indexes, plan_indexes[7*base_size:8*base_size], base_size)
    transpose_id = 8

    for streak_len in off_diagonal_streak_lens:
        for off_diagonal_id in range(streak_len):
            data_transpose_off_diagonal(
                indexes,
                plan_indexes[(transpose_id)*base_size:(transpose_id+1)*base_size],
                plan_indexes[(transpose_id+1)*base_size:(transpose_id+2)*base_size],
                base_size
            )
            data_transpose_off_diagonal(
                indexes,
                plan_indexes[(transpose_id+2)*base_size:(transpose_id+3)*base_size],
                plan_indexes[(transpose_id+3)*base_size:(transpose_id+4)*base_size],
                base_size
            )
            transpose_id += 4

        data_transpose_diagonal(indexes, plan_indexes[(transpose_id)*base_size:(transpose_id+1)*base_size], base_size)
        data_transpose_diagonal(indexes, plan_indexes[(transpose_id+1)*base_size:(transpose_id+2)*base_size], base_size)
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[(transpose_id+2)*base_size:(transpose_id+3)*base_size],
            plan_indexes[(transpose_id+3)*base_size:(transpose_id+4)*base_size],
            base_size
        )
        data_transpose_off_diagonal(
            indexes,
            plan_indexes[(transpose_id+4)*base_size:(transpose_id+5)*base_size],
            plan_indexes[(transpose_id+5)*base_size:(transpose_id+6)*base_size],
            base_size
        )
        data_transpose_diagonal(indexes, plan_indexes[(transpose_id+6)*base_size:(transpose_id+7)*base_size], base_size)
        data_transpose_diagonal(indexes, plan_indexes[(transpose_id+7)*base_size:(transpose_id+8)*base_size], base_size)

        transpose_id += 8

    return indexes


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

def interleave(x):
    y = np.array(x)
    half_len = len(x) // 2
    for i in range(half_len):
        y[2*i] = x[i]
        y[2*i + 1] = x[i + half_len]
    return y

def deinterleave(x):
    y = np.array(x)
    half_len = len(x) // 2
    for i in range(half_len):
        y[i] = x[2*i]
        y[i + half_len] = x[2*i + 1]
    return y

def extend(arr, times):
    new_arr = np.zeros(arr.size * times, arr.dtype)

    for i in range(arr.size):
        for j in range(times):
            new_arr[times * i + j] = arr[i]

    return new_arr

def repeat(arr, times):
    new_arr = np.zeros(arr.size * times, arr.dtype)

    for j in range(times):
        for i in range(arr.size):
            new_arr[arr.size *  j + i] = arr[i]

    return new_arr

def every_jump_forward(arr, jump):
    new_arr = 0.* arr

    for i in range(jump):
        for j in range(arr.size // jump):
            new_arr[arr.size//jump*i + j] = arr[j * jump + i]

    return new_arr


def every_jump_backward(arr, jump):
    new_arr = 0.* arr

    for i in range(jump):
        for j in range(arr.size // jump):
            new_arr[j * jump + i] = arr[arr.size//jump*i + j]

    return new_arr


def small_fft_2b(signal, small_len = 1):
    signal_len = len(signal)

    signal =  np.array(signal, dtype = np.complex_)

    scrambled_indexes_ = scrambled_indexes(signal_len)
    signal = scrambled_signal(scrambled_indexes_, signal, 1.)
    small_scrambled_indexes_ =  scrambled_indexes(small_len)



    n_small_fft = signal_len // small_len

    # for i in range(signal_len // small_len):
    #     mini_signal = signal[small_len * i : small_len * (i + 1)]
    #     mini_signal = scrambled_signal(small_scrambled_indexes_, mini_signal, 1.)
    #     signal[small_len * i : small_len * (i + 1)] = mini_signal

    subtwiddle_len = 1



    twiddles = []
    while subtwiddle_len < small_len:
        twiddles.append (
            repeat(
                myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len),
                small_len // subtwiddle_len // 2
            )
        )
        subtwiddle_len *= 2

    work_signal_a = np.array(signal, dtype = np.complex_)
    work_signal_b = np.array(signal, dtype = np.complex_)


    subtwiddle_len = 1
    jump = subtwiddle_len // small_len
    twiddles_id = 0
    while subtwiddle_len < small_len:
        for i in range(n_small_fft):
            for j in range(small_len // 2):
                # this part is SIMD parallizable
                a_id = i * small_len + j
                b_id = i * small_len + j + small_len // 2

                tw_b_id = j

                work_signal_b[b_id] = (
                    work_signal_b[b_id]
                    * twiddles[twiddles_id][tw_b_id]
                )


                work_signal_a[a_id] = (
                    work_signal_b[a_id]
                    + work_signal_b[b_id]
                )

                work_signal_a[b_id] = (
                    work_signal_b[a_id]
                    - work_signal_b[b_id]
                )
            if (twiddles_id < len(twiddles) - 1):
                work_signal_a[i * small_len:(i+1) * small_len] = interleave( work_signal_a[i * small_len:(i+1) * small_len])
        twiddles_id += 1
        subtwiddle_len *= 2
        print(work_signal_a)
        work_signal_b = work_signal_a * 1.


    for i in work_signal_a:
        print(i)

    return work_signal_a


def prepare_radix4(size, base_len, signal, spectrum, stride, signal_start = 0, spectrum_start = 0):
    if size == base_len:
        for i in range(size):
            spectrum[i] = signal[i * stride]
            print(spectrum_start + i, signal_start + i*stride)
    else:
        for i in range(2):
            prepare_radix4(
                size // 2,
                base_len,
                signal[i * stride:],
                spectrum[i * (size // 2):],
                stride * 2,
                signal_start + i * stride,
                spectrum_start + i * (size // 2)
            )
def lzcnt64(v):
    d = np.double(v & ~(v >> 1))
    return (1086 - ((d.view(np.int64) >> 52))) - (v == 0) * 1022

def autobit_reverse(n):
    n = np.int64(n)
    active_bits = (n - 1)
    incrementer = active_bits ^ (n >> 1)
    positive_mask = ~(np.int64(1) << 63)

    all_64_bits = ~np.int64(0)

    v = active_bits

    for i in range(n >> 1):
        print("out_real[", n-1- 2*i, "] = in_real[", v, "];")

        # go to next value
        v = v & incrementer

        print("out_real[", n-2- 2*i, "] = in_real[", v, "];")


        lzcnt = lzcnt64(v)

        # clear / traverse up tree
        v = (v << lzcnt) & positive_mask  # dow
        v = v >> lzcnt

        # traverse down tree
        traverse_down = all_64_bits << (64 - lzcnt)
        traverse_down = traverse_down & active_bits
        v = v | traverse_down


#prepare_radix4(64,2,inp,out,1)

def small_fft_2(signal, small_len = 1):
    signal_len = len(signal)

    signal =  np.array(signal, dtype = np.complex_)

    subtwiddle_len = 1

    twiddles = []
    scrambled_indexes_ = scrambled_indexes(signal_len)
    signal = scrambled_signal(scrambled_indexes_, signal, 1.)
    while subtwiddle_len < small_len:
        twiddles.append (
            extend(
                myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len),
                small_len // subtwiddle_len // 2
            )
        )
        subtwiddle_len *= 2

    work_signal_a = np.array(signal, dtype = np.complex_)
    work_signal_b = np.array(signal, dtype = np.complex_)

    n_small_fft = signal_len // small_len

    subtwiddle_len = 1
    twiddles_id = 0
    while subtwiddle_len < small_len:

        for i in range(n_small_fft):
            work_signal_b[i*small_len : (i+1)*small_len] = deinterleave(work_signal_a[i*small_len : (i+1)*small_len])
            for j in range(small_len//2):
                # this part is SIMD parallizable
                work_signal_b[small_len * i + j + small_len // 2] = (
                    work_signal_b[small_len * i + j + small_len // 2]
                    * twiddles[twiddles_id][j]
                )


                work_signal_a[small_len * i + j] = (
                    work_signal_b[small_len * i + j]
                    + work_signal_b[small_len * i + j + small_len // 2]
                )

                work_signal_a[small_len * i + j + small_len // 2] = (
                    work_signal_b[small_len * i + j]
                    - work_signal_b[small_len * i + j + small_len // 2]
                )

        print(work_signal_a)
        twiddles_id += 1
        subtwiddle_len *= 2


    # subtwiddle_len = 1
    # while subtwiddle_len < small_len:
    #     work_signal_a = interleave(work_signal_a)
    #     subtwiddle_len*=2

    for i in work_signal_a:
        print(i)

    return work_signal_a



def small_fft_4(signal, small_len = 1):
    signal_len = len(signal)

    subtwiddle_len = 1

    twiddles = []
    scrambled_indexes_ = scrambled_indexes(signal_len)
    signal = scrambled_signal(scrambled_indexes_, signal, 1.)

    while (subtwiddle_len * 4) <= small_len:
        twiddles.append(
            extend(
                np.concatenate((
                    myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len),
                    myexp(np.arange(0, subtwiddle_len, dtype=float) / 2 / subtwiddle_len),
                    myexp(np.arange(0, subtwiddle_len, dtype=float) * 1.5 / subtwiddle_len )
                )),
                signal_len // subtwiddle_len // 4
            )
        )
        subtwiddle_len *= 4

    while subtwiddle_len < small_len:
        twiddles.append (
            extend(
                myexp(np.arange(0, subtwiddle_len, dtype=float) / subtwiddle_len),
                signal_len // subtwiddle_len // 2
            )
        )
        subtwiddle_len *= 2

    work_signal_a = signal
    work_signal_b = signal.copy()

    subtwiddle_len = 1
    twiddles_id = 0
    while (subtwiddle_len * 4) <= small_len:
        work_signal_b = deinterleave(work_signal_a)
        work_signal_a = deinterleave(work_signal_b)
        work_signal_b = work_signal_a.copy()
        print(work_signal_b)
        for i in range(signal_len // 4 // small_len):
            for j in range(small_len):
                # this part is SIMD parallizable
                a_id = small_len * i + j
                b_id = small_len * i + j + signal_len // 4
                c_id = small_len * i + j + signal_len // 2
                d_id = small_len * i + j + 3 * signal_len // 4

                tb_id = a_id
                tc_id = b_id
                td_id = c_id

                work_signal_b[b_id] = (
                    work_signal_b[b_id]
                    * twiddles[twiddles_id][tb_id]
                )

                work_signal_b[c_id] = (
                    work_signal_b[c_id]
                    * twiddles[twiddles_id][tc_id]
                )

                work_signal_b[d_id] = (
                    work_signal_b[d_id]
                    * twiddles[twiddles_id][td_id]
                )


                work_signal_a[a_id] = ( #done
                    work_signal_b[a_id]
                    + work_signal_b[b_id]
                )

                work_signal_a[b_id] = (
                    work_signal_b[a_id]
                    - work_signal_b[b_id]
                )
                work_signal_a[c_id] = ( #done
                    work_signal_b[c_id]
                    + work_signal_b[d_id]
                )
                work_signal_a[d_id] = ( #done
                    work_signal_b[c_id]
                    - work_signal_b[d_id]
                )


                # Second butterfly.
                work_signal_b[a_id] = ( #done
                    work_signal_a[a_id]
                    + work_signal_a[c_id]
                )

                work_signal_b[b_id] = (
                    work_signal_a[b_id]
                    - 1j* work_signal_a[d_id]
                )

                work_signal_b[c_id] = (
                    work_signal_a[a_id]
                    -  work_signal_a[c_id]
                )

                work_signal_b[d_id] = (
                    work_signal_a[b_id]
                    + 1j * work_signal_a[d_id]
                )

        work_signal_a = work_signal_b

        twiddles_id += 1
        subtwiddle_len *= 4



    while subtwiddle_len < small_len:
        work_signal_b = deinterleave(work_signal_a)
        for i in range(signal_len // 2 // small_len):
            for j in range(small_len):
                # this part is SIMD parallizable
                work_signal_b[small_len * i + j + signal_len // 2] = (
                    work_signal_b[small_len * i + j + signal_len // 2]
                    * twiddles[twiddles_id][small_len * i + j]
                )

                work_signal_a[small_len * i + j] = (
                    work_signal_b[small_len * i + j]
                    + work_signal_b[small_len * i + j + signal_len // 2]
                )

                work_signal_a[small_len * i + j + signal_len // 2] = (
                    work_signal_b[small_len * i + j]
                    - work_signal_b[small_len * i + j + signal_len // 2]
                )

        twiddles_id += 1
        subtwiddle_len *= 2



    subtwiddle_len = 1
    while subtwiddle_len < small_len:
        work_signal_a = interleave(work_signal_a)
        subtwiddle_len*=2

    for i in work_signal_a:
        print(i)

    return work_signal_a


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
        #
        # print(self.dft_mat)
        # print(self.dft_mat_t)


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

        print(work_signal_a)

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

        for i in work_signal_a:
            print(i)

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

        twiddle_id =  len(self.twiddles) - 1 # TODODIF

        print("copy", work_signal_a)

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

        print("copy", work_signal_a)

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

        print("copy", work_signal_a)

        # #specialty
        # if self.dft_len != 1:
        #     subfft_len *= self.dft_len
        #     n_subfft_len //= self.dft_len
        #
        #     a_id = 0
        #     bb_id = 0
        #     for subfft_id in range(n_subfft_len):
        #         for dft_basis_id in range(self.dft_len):
        #             # dft_basis_id = 0 specialty, = 1 for all
        #             # dft_basis_id = 1 specialty, = real for all
        #             b_id = bb_id
        #             M = work_signal_a[a_id]
        #             dft_basis = self.dft_mat[dft_basis_id]
        #             for dft_factor_id in range(self.dft_len):
        #                 work_signal_b[b_id] += (
        #                     M
        #                     *  dft_basis[dft_factor_id]
        #                 )
        #                 b_id += 1
        #             a_id += 1
        #         bb_id += self.dft_len
        #
        #
        #     work_signal_a = work_signal_b
        #     work_signal_b = work_signal_a.copy()
        #
        # twiddle_id = 0

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
                    # dft_basis_id = self.dft_len/2 specialty, = real for all
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


sig_len =  8

a = random_complex(sig_len)
b = random_complex(sig_len)

r = random_complex(sig_len)
for k in range(0, sig_len, 2):
    r[k] = (k//2) + ((k//2) & 1)*1.j
    r[k+1] = -1 - k//2 + ((k//2) & 1)*1.j

myfft = Myfft(sig_len,1)
def mfft(r):
    return myfft.process(r, False)

def mifft(r):
    return myfft.process(r, True)

def mfft_nb(r):
    return myfft.process_dif(r, False, True)

def mifft_nb(r):
    return myfft.process(r, True, True)

# f = ifft(fft(a) * fft(b))
# m1 = mifft(mfft(a) * mfft(b))
# m2 = mifft_nb(mfft_nb(a) * mfft_nb(b))
# myfft.process_dif(r)
n = 32
s = 8
(1+0j) * np.array(range(n))
nn = np.zeros(n)
nn[8] = 1.
print("-------------")
small_fft_2((1+0j) * np.array(range(n)), s)

# print("-------------")
# small_fft_2b(nn, s)

print("-------------")
# small_fft_4((1+0j) * np.array(range(n)), s)
# print("-------------")
myfft = Myfft(n,s)
# print("-------------")
myfft.process((1+0j) * np.array(range(n)))
#
# print(np.max(np.abs(f-m1)), "f-m1")
# print(np.max(np.abs(f-m2)), "f-m2")

