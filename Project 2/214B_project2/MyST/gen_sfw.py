# import os

f = open('train_vtlp_sfw_sp_pp.csv', 'r')

g = open('train_vtlp_sfw_sp_pp_new.csv', 'w')


for line in f:

    l_n = line.replace('data_vtlp_sp_1.1_0.9', 'data')

    g.write(l_n)

f.close()
g.close()

# h = open('sfw.csv', 'r')
# g = open('sfw_new.csv', 'a')

# for i,line in enumerate(h):

#     if i==0:
#         continue

#     # l_n = line.replace('data/MyST/', 'data_vtlp_sp_1.1_0.9/MyST/')

#     g.write(line)

# h.close()
# g.close()   

# import torch
# import torchaudio
# import torchaudio.transforms as T
# import soundfile as sf
# import random
# import librosa
# import numpy as np
# from matplotlib import pyplot as plt
# import os
# from tqdm import tqdm
# def get_spectrogram(wav, n_fft=512, win_len=25, hop_len=10,power=2.0):
#     spectrogram = T.Spectrogram(
#         n_fft=n_fft,
#         win_length=win_len,
#         hop_length=hop_len,
#         center=True,
#         pad_mode="reflect",
#         power=power,
#     )
#     return spectrogram(wav)

# def source_extract(spec, gamma=0.2):

#     spec = spec.squeeze(0)
#     env = torch.zeros_like(spec)

#     env[0,:] = spec[0,:]
#     for i in range(1, env.size(0)):
#         env[i,:] = torch.maximum(env[i-1,:], spec[i,:])
#         env[i,:] = torch.maximum(env[i,:], gamma*(env[i-1,:] - spec[i,:]))
    
#     for i in range(env.size(0)-2,-1,-1):
#         env[i,:] = torch.maximum(env[i+1,:], spec[i,:])
#         env[i,:] = torch.maximum(env[i,:], gamma*(env[i+1,:] - spec[i,:]))
    
#     #implement backward pass

#     eps = 1e-5

#     env[env==0] = eps

#     source = spec/env

#     # print((env == 0).nonzero())

#     # print("og spec", torch.isnan(spec).any())
#     # print("og source ",torch.isnan(source).any())
#     # print("og env",torch.isnan(env).any())
    
#     return source, env

# def vtlp(spec, factor, sr):

#     freq_dim, time_dim = spec.shape

#     def get_scale_factors(freq_dim, sampling_rate=16000, fhi=4800, alpha=0.9):
#         factors = []
#         freqs = np.linspace(0, 1, freq_dim)

#         scale = fhi * min(alpha, 1)
#         f_boundary = scale / alpha
#         half_sr = sampling_rate / 2

#         for f in freqs:
#             f *= sampling_rate
#             if f <= f_boundary:
#                 factors.append(f * alpha)
#             else:
#                 warp_freq = half_sr - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - f)
#                 factors.append(warp_freq)

#         return np.array(factors)

#     factors = get_scale_factors(freq_dim, sampling_rate = sr, alpha=factor)
#     factors *= (freq_dim - 1) / max(factors)
#     new_spec = torch.zeros_like(spec)

#     for i in range(freq_dim):
#         # first and last freq
#         if i == 0 or i + 1 >= freq_dim:
#             new_spec[i, :] += spec[i, :]
#         else:
#             warp_up = factors[i] - np.floor(factors[i])
#             warp_down = 1 - warp_up
#             pos = int(np.floor(factors[i]))

#             new_spec[pos, :] += warp_down * spec[i, :]
#             new_spec[pos+1, :] += warp_up * spec[i, :]

#     return new_spec

# def plot_spectrogram(spec,aug,  alpha, beta, aug_title="Shifted", og_title = "Original", ylabel="freq_bin", aspect="auto", xmin=8000, xmax=10000):
#     fig, axs = plt.subplots(2, 1)
#     og_title = og_title + " alpha = " +str(alpha) + ", beta = "+str(beta)
#     axs[0].set_title(og_title)
#     axs[0].set_ylabel(ylabel)
#     axs[0].set_xlabel("frame")
#     im = axs[0].imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
#     if xmax:
#         axs[0].set_xlim((xmin, xmax))
#     axs[1].set_title(aug_title)
#     axs[1].set_ylabel(ylabel)
#     axs[1].set_xlabel("frame")
#     im = axs[1].imshow(librosa.power_to_db(aug), origin="lower", aspect=aspect)
#     if xmax:
#         axs[1].set_xlim((xmin, xmax))
#     # fig.colorbar(im, ax=axs)
#     fig_str = "Spectrogram.png"
#     plt.show(block=False)
#     plt.savefig(fig_str)


# def sfw(p):

#     path  = '/data/balaji/Project2_Group1/214B_project2/Librispeech/' + p
#     audio, sr = torchaudio.load(path)

#     power = get_spectrogram(audio)
#     source, filt = source_extract(power)
    
#     alpha = random.uniform(1.0, 1.3)
#     beta = random.uniform(1.0, 1.3)

#     # alpha = 1.3
#     # beta = 1.3
#     source_aug = vtlp(source, alpha, sr)

#     filt_aug = vtlp(filt, beta, sr)

#     spec_aug = source_aug*filt_aug


#     # print("Power", torch.isnan(power).any())
#     # print("Source ",torch.isnan(source).any())
#     # print("Filter",torch.isnan(filt).any())
#     # print("Source VTLP",torch.isnan(source_aug).any())
#     # print("Filter VTLP",torch.isnan(filt_aug).any())
#     # print("Spec VTLP",torch.isnan(spec_aug).any())

#     try:
#         audio_aug = librosa.griffinlim(np.array(spec_aug), n_iter = 8, hop_length=10, win_length=25)
#     except:
#         plot_spectrogram(torch.abs(power[0]),torch.abs(spec_aug), alpha, beta, aspect="equal")
#         return None
#     # plot_spectrogram(torch.abs(power[0]),torch.abs(spec_aug), alpha, beta, aspect="equal")

#     # new_path = '/data/balaji/workdir/trials/egs/libri_aug/data/train_clean_100/' + '/'.join(path.split('/')[6:])

#     # print(new_path)

#     p_n = p.replace('data/LibriSpeech/librispeech_finetuning/1h', 'data_vtlp_sp_1.1_0.9/MyST/train/sfw')

#     path_n = '/data/balaji/Project2_Group1/214B_project2/MyST/' + p_n

#     # access = 0o777



#     os.makedirs(os.path.dirname(path_n), exist_ok=True)

#     sf.write(path_n, audio_aug, sr)

#     return p_n

# import os

# f = open('/data/balaji/Project2_Group1/214B_project2/Librispeech/train_1h.csv', 'r')

# g = open('/data/balaji/Project2_Group1/214B_project2/MyST/sfw.csv', 'w')

# for i,line in tqdm(enumerate(f)):

#     if i==0:
#         g.write(line)
#         continue


#     # l_n = line.replace('data/MyST/', 'data_vtlp_sp_1.1_0.9/MyST/')

#     text, path, audio = line.split(',') 

#     path_n = sfw(path)

#     l_n = ','.join([text,path_n,path_n])+'\n'


#     g.write(l_n)

# f.close()
# g.close()

