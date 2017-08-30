Thank you for this manuscript, Martin. It's awesome. Very easy to follow &mdash; the level is just right. And the illustrations are really nice.

Since we're a little short of time, I have made some corrections directly in the manuscript...

https://github.com/seg/tutorials-2017/blob/master/1710_Coloured_inversion/Manuscript-edit-Matt.ipynb

I realize this makes it a little hard to diff, especially in a notebook. Possibly you could `ipython nbconvert --to markdown` both of them and diff them that way.

Main things I did:

- Minor spelling and grammar corrections. Nothing major. (I use 'spectrum' and 'spectrums' for the singular and plural, respectively. Maybe this is weird to some people, but I prefer it to the use of 'spectra' for both.)
- Some alterations to the code snippets. This is mainly either PEP8 or because the columns are only about 50-55 characters wide, so we cannot have long lines. I also thought you might like `np.apply_over_axis()` instead of the generator expression for the convolution. I think it's faster too.
MB: *Thank you for that!*
- Changed some 'house style' things, eg layout, references, etc. Nothing major.
- I made the subheads match the list of steps in the algorithm, including their numbers.
- I cut a few sentences that felt like 'details' to me. We can leave these details in the notebook in the GitHub repo &mdash; we often do this.

I have some suggestions for you:

- Please read through and check my edits!
- Figure 1: Do you think it's worth adding the well?
MB: *Not sure, it's on the next figure.*
- In `T_log = 1000 / depth_f021` I'm not sure what's happening. Have you thought about using the DT log to get velocity then using interpolation to switch the logs to regularly sampled time?
MB: *I used the DT log to convert to time using interpolation but did not resample it. Since I am approximating with a function later, it's not necessary I think. However, I did not change the name of the variable which makes it confusing. I also think the shortcut I was taking earlier (not in the snippet) was wrong and altered the resulting spectrum (did not change the end result...). I was working with constant time step in my previous attempts at CI...maybe this was a better idea. Anyway, I edited the snippet and the figures accordingly).*

- Figure 2: The TLE staff always remove references to (a) and (b) in the figure itself. I recommend changing the plot titles to "Impedance" and "Log spectrum". If you can, move them inside the frame of the plot, in the upper left (to save a bit of vertical space). They look good if a little larger and bold.
MB: *Done*

- Figure 3: I don't think this plot needs a title; the caption explains.
MB: *Title removed*

- Figure 3: (and throughout) I recommend using a different *shade* of colour for the regression line, because this one will be hard to see in greyscale. Perhaps lighter points and darker line makes most sense. Please use the same colours you choose for the same elements in every figure.
MB: *Let me know if this color_palette works*
- Figure 4: I wonder if it's worth boosting the amplitude of the seismic a bit, since it's arbitrary anyway? Just to make it a little closer to the others (but without clashing with it). Anything you can do to make the figure less tall will help with layout.
MB: *I think you are right. This looks better with the boosted seismic amplitude*
- In the code block where you generate the operator, can you show where `gap` comes from? MB: *Gap is just the substraction of the seismic_spectrum from the approximated log spectrum. I edited the snippet.* Also, it seems like `fftshift` is operating on a time-domain signal, is it not? So this might be better described as a time shift? Or maybe I got lost...
MB: *Please check if this is clear (added in the text):"and shift the zero frequency component of the operator to the center of the spectrum". Since I am taking the imaginary part after, It was like a phase rotation. Maybe a bit of a shortcut to say that anyway...*
- Figure 5: I don't think this is the difference spectrum, is it?
MB: *Yes it is. With gap, I just substract 10 from 10000 (looks larger but, it is in log space). TBH it is still one part of the workflow that depends on wether you scale the seismic spectrum or not. I rescaled it for plotting purpose, and kept using it for the gap and the operator. The difference is now more obvious, but it has little impact on the end result.*
- Figure 5: As before, please either move the plot titles or put them inside the plot frames.
MB: *Done*
- Figure 6: Nice figure... you could make it a little wider I think, maybe 25-50% wider. Also, can you put the colourbar inside the frame? Or, if that's too fiddly, put it on the right. Again, this will help with the layout.
MB:*Edited, let me know if you need more changes*
- QC step... "this fit was achieved with a manual fit"... not sure I follow. You mean because we chose the upper and lower tapers? Maybe we can say "there are some variables", and you went through some iterations to optimize them a bit?
MB: *There are some things you can play with, the taper, the normalization, the number of traces to calculate your spectra. I guess that was just me saying, if I had to do that for production, I would minimize some kind of function base on different constraints on those variables. I will just remove that sentence. If you feel like saying something about that, I am on board with whatever you think is best*

If you can add the 'full' Notebook to this repo, I'm happy to help with some of this, if I can.
MB: *Still a lot of French and redundancy in this one. If it not necessary for now, let me clean it and I will supply one soon. I don't have time to do that before your deadline.*

I understand you cannot get to this till Tuesday. If we can get these changes made by end of day on Wednesday, I think we're still in good shape.

Thank you again for this brilliant contribution, Martin! I think people are really going to love it.

Matt
