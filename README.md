# NOAA APT DECODER

A learning experiment in decoding NOAA APT signals "from scratch." 

My DSP background is weak, so I'm still working to better understand the mechanics
of the Hilbert transform to extract instantaneous amplitude information.

# Sample use

A sample of the CLI usage using Fire:
```
python noaa_apt_decoder.py --in_path=06261434.wav --out_path image.png --black_point 5 --white_point 95 --components ["image_a","image_b"]
```
![](https://raw.githubusercontent.com/peterbbryan/NOAA-APT-decoder-experiments/master/sample_output.png)

```
python noaa_apt_decoder.py --in_path 06261434.wav --out_path temp.png --black_point 5 --white_point 70 --components ["image_a"] --colorize True
```
![](https://raw.githubusercontent.com/peterbbryan/NOAA-APT-decoder-experiments/master/sample_colorized.png)

# Special thanks
I found the following resources invaluable:
* https://noaasis.noaa.gov/NOAASIS/pubs/Users_Guide-Building_Receive_Stations_March_2009.pdf
* https://noaa-apt.mbernardi.com.ar/how-it-works.html 
* https://github.com/zacstewart/apt-decoder
