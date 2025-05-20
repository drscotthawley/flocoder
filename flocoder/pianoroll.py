#!/usr/bin/env python3

"""
various routines for converting midi files to piano roll images and back
Unless otherwise noted: Author: Scott H. Hawley, Feb-March 2024
"""

import os
import torch 
import torchvision 
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
#from .utils import rect_to_square, square_to_rect
#from .chords import chord_num_to_color, simplify_chord, CHORD_BORDER
CHORD_BORDER = 1  # chord borders never worked well, so just set to 1 pixel for safety
ONSET_STYLE = 'start'  # 'early'=onset markers on pixels before notes, 'start'=onset markers are part of notes



def square_to_rect(img):
    #"""just an alias for flip_bottom_half_and_attach"""
    return flip_bottom_half_and_attach(img)

def rect_to_square(img):
    "takes a 512x128 image and returns a 256x256 image with the bottom half reversed"
    w, h = img.size
    new_img = Image.new(img.mode, (w//2, h*2))
    new_img.paste(img.crop((0, 0, w//2, h)), (0, 0))
    new_img.paste(img.crop((w//2, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT), (0, h))
    return new_img


def plot_piano_roll(pr_array): 
    plt.figure(figsize=(8, 8))
    plt.imshow(np.flipud(pr_array), aspect='auto')
    plt.show()

def piano_roll_to_pretty_midi(piano_roll, fs=8, program=0):
    # this routine copied from https://github.com/jsleep/pretty-midi/blob/ba7d01e5796fedf3ca0a3528e48b5242d9d2ccc3/examples/reverse_pianoroll.py
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    #print("piano_roll.T[piano_roll.T != 0] = ",piano_roll.T[piano_roll.T != 0],flush=True)
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = np.clip(piano_roll[note, time + 1], 0, 127)
        #print("piano_roll[note, time + 1], velocity = ",piano_roll[note, time + 1], velocity,flush=True)
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



#### beginning of code copied from midi2img.py

def find_first_note_start(midi):
    """find the start time of the first note in the midi file
       used to help alignment to beats/bars
    """
    first_start = 10000.0
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start < first_start:
                first_start = note.start
    return first_start


def get_piano_rolls(midi, fs, remove_leading_silence=True, add_onsets=True, debug=False):
    """Converts a pretty_midi object to a piano roll for each instrument"""
    duration = midi.get_end_time()   # find out duration of the midi file
    n_frames = int(np.ceil(duration * fs))   # calculate the number of frames

    # create a piano roll for each instrument
    # TODO: currently this is only setup for POP909 dataset, need to generalize for other datasets
    piano_rolls = {'PIANO':  np.zeros((128, n_frames)),
                   'MELODY': np.zeros((128, n_frames)),
                   'TOTAL':  np.zeros((128, n_frames))}
    if remove_leading_silence:
        first_start = find_first_note_start(midi)

    for instrument in midi.instruments: 
        name = instrument.name.upper()
        if name in ['MELODY', 'PIANO']:
            if debug: print(f"get_piano_rolls: instrument.name = {name}")
            for note in instrument.notes:
                if remove_leading_silence:
                    note.start -= first_start
                    note.end -= first_start
                start = int(np.round(note.start * fs))  # quantize start time to nearest 16th note
                dur = (note.end - note.start)*fs        # quantize duration  (Tip: don't separately quantize start & end; that can lead to "double-rounding" errors)
                #end = int(np.round(note.end * fs)) 
                end = start + int(np.round(dur))   # round means some notes will get held a bit too long, but "floor" would err on the side of extra staccatto notes which I don't want
                if end==start: end = start+1  # make sure note is at least 1 pixel long
                piano_rolls[name][note.pitch, start:end] = note.velocity ## value of piano roll array for these pixels will be the note velocity. end+1 so that "end" index gets covered
                piano_rolls['TOTAL'][note.pitch, start:end] = note.velocity
                #if note.velocity in [65,59,49,100]: print("note = ",note)

                # extra fun: make sure all note onsets pop
                piano_rolls[name][note.pitch, start-1] = 0
                piano_rolls['TOTAL'][note.pitch, start-1] = 0

    # if remove_leading_silence and add_onsets: # we need to add one pixel for the red onset dot at the start
    #     for instrument in piano_rolls:
    #         piano_rolls[instrument] = np.insert(piano_rolls[instrument], 0, 0, axis=1)

    return piano_rolls


def piano_roll_to_img(pr_frame, # this is an array of shape (128, n_frames)
                      output_dir, midi_name, instrument, 
                      start_col=None, add_onsets=True, chords=None, chord_names=False, debug=False,
                      onset_style=ONSET_STYLE, # 'start' or 'early'
                      ):
    os.makedirs(f"{output_dir}/{midi_name}", exist_ok=True)
    filename = f"{output_dir}/{midi_name}/{midi_name}_{instrument}.png"
    if start_col is not None: 
        filename = filename.replace(".png",f"_{str(start_col).zfill(5)}.png") 
    #if debug: print("pr_frame.T[pr_frame.T != 0] = ",pr_frame.T[pr_frame.T != 0])
    #scaling_factor = 65 / 18  # found empiracally by lots of checking
    #pr_frame = np.round(pr_frame * scaling_factor).astype(np.uint8)

    scale_factor = 2  # velocity only goes up to 127, but colors go up to 255
    green_channel = np.clip(np.round(pr_frame*scale_factor), 0, 255).astype(np.uint8)
    rgb_image = np.dstack((np.zeros_like(green_channel), green_channel, np.zeros_like(green_channel)))
    img = Image.fromarray(rgb_image,'RGB')

    if add_onsets:        # add little onset markers (red dots)
        if onset_style=='early':
            # any black pixel that has a green pixel to its right is an onset. color it red
            # note that x any are flipped from what you'd think, e.g. "img.size =  (2352, 128)"
            for y in range(img.size[-1]):
                for x in range(img.size[0]-1):
                    if img.getpixel((x,y)) == (0,0,0) and img.getpixel((x+1,y)) != (0,0,0):
                        img.putpixel((x,y), (255,0,0)) 
        elif onset_style=='start':
            # New version: 
            # any green pixel with a black pixel on its left becomes a red pixel. or if first pixel on row is green, make it red (matchinf velocity)
            # Thus red pixel counts as both onset and first part of note, so shortest notes (16ths) will appear as only red with no green
            #   btw this seems to agree w/ polyffusion's approach (??)
            for y in range(img.size[-1]):
                x = 0
                pxl = img.getpixel((x,y))
                r,g,b = pxl
                if is_green(*pxl):
                    img.putpixel((0,y), (g,0,0))  # make the first pixel of the note red, matching the green intensity
                for x in range(1, img.size[0]):
                    pxl = img.getpixel((x,y))
                    r,g,b = pxl
                    if is_green(*pxl) and is_black(*img.getpixel((x-1,y))):
                        img.putpixel((x,y), (g,0,0))
        else:
            print(f"Error: Unrecognized onset_style = {onset_style}. Exiting.")
            return

    img = img.transpose(Image.FLIP_TOP_BOTTOM) # flip it vertically for display purposes


    if chords is not None: # add the chord colors for each time as a rectangles at the top and bottom
        if chord_names:
            font_size = 7
            try:
                myFont = ImageFont.truetype("arial.ttf", 7) #mac
            except:
                myFont = ImageFont.load_default(size=font_size)


        for c in chords:
            color = chord_num_to_color(c['chord_num'])
            img.paste(color, (int(c['start']), img.size[-1]-CHORD_BORDER, int(c['end']), img.size[-1]))
            img.paste(color, (int(c['start']), 0, int(c['end']), CHORD_BORDER))
            if chord_names:
                chord_name = c['chord_name'].replace(':','')
                if debug: print(f"chord_name = {chord_name}, chord_num = {c['chord_num']}")
                xpos = int(c['start'])
                I1 = ImageDraw.Draw(img)
                I1.text((xpos, 0), chord_name, font=myFont, fill=(255, 255, 255))

    if debug: print("img.size = ",img.size)
    if 0 in img.size:
        print(f"Error: img.size = {img.size}. Skipping this file.")
        return
    
    # # just make sure all blue is gone: 
    # img_array = np.array(img)
    # img_array[:, :, 2] = 0
    # img = Image.fromarray(img_array)
    
    img.save(filename)



def check_for_melody_piano(midi: pretty_midi.PrettyMIDI, debug=False):
    has_melody, has_piano = False, False
    if debug:
        print("check_for_melody_piano: midi.instruments = ",midi.instruments)
    for i, instrument in enumerate(midi.instruments):
        if debug: print(f"check_for_melody_piano: instrument = [{instrument.name.upper()}]")
        if instrument.name.upper() == 'MELODY': has_melody = True
        if instrument.name.upper() == 'PIANO':  has_piano = True
    # if theres only one instrument with no name, name it PIANO
    if len(midi.instruments) == 1 and midi.instruments[0].name == '':
        has_piano = True
        midi.instruments[0].name = 'PIANO'
    return has_melody, has_piano



def midi_to_pr_img(midi_file, output_dir, 
                   show_chords=None, # to show chords or not
                   all_chords=None,  # list of all possible chords
                   add_onsets=True,  # add red dots for note onsets
                   chord_names=False, # to show chord names or not
                   filter_mp=True,   # filter midi & piano
                   remove_leading_silence=True, # remove silence at start of song
                   simplify_chords=False, # simplify chord names
                   debug=False,):    # show debugging info 
    """Converts a MIDI file to a piano roll image"""
    if debug: print(f"midi_to_pr_img: Processing {midi_file}")
    if '/versions/' in midi_file and args.skip_versions: return
    midi = pretty_midi.PrettyMIDI(midi_file)

    if not check_for_melody_piano(midi):
        print(f"Not ok: File {midi_file} does not have melody and piano. Skipping")
        return
    else:
        if debug: print(f"Ok: File {midi_file} has melody and piano. Processing") 

    ### Normalize tempo to 120bpm
    tempo_changes = midi.get_tempo_changes()
    start_tempo = tempo_changes[1][0]
    bps = start_tempo / 60.0
    fs = bps * 4.0 * 2
    if debug: print("start_tempo, fs = ", start_tempo, fs)

    chords=None
    if show_chords and all_chords is not None:
        # read the chord timing file, but note that those times have not yet been normalized to 120bpm 
        # this file has column-separated format "start_time end_time chord"
        chords_path = midi_file.replace('.mid', '_chords.txt')
        with open(chords_path) as f:
            chords = f.read().splitlines()
        # split each line of text into a dict 3 values {'start':, 'end':, 'chord':}:
        chords = [dict(zip(['start', 'end', 'chord'], c.split('\t'))) for c in chords]

        for c in chords:
            c['start'] = int(np.floor(float(c['start']) * fs))
            c['end'] = int(np.ceil(float(c['end']) * fs))
            c['chord_name'] = simplify_chord(c['chord']) if simplify_chords else c['chord']     
            c['chord_num'] = all_chords.index(c['chord_name'])              

    if filter_mp:  # remove non-piano, non-melody instruments
        midi.instruments = [instrument for instrument in midi.instruments if instrument.name.upper() in ['MELODY', 'PIANO']]
  
    piano_rolls = get_piano_rolls(midi, fs, remove_leading_silence=remove_leading_silence, add_onsets=add_onsets)
    if debug:
        for p in piano_rolls.keys():
            print(f"p  {p}.shape =",piano_rolls[p].shape)
            #print(f"piano_rolls[{p}][piano_rolls[p] != 0] = ",piano_rolls[p][piano_rolls[p] != 0])
 
    midi_name = os.path.basename(midi_file).split('.')[0] # get the midi filename w/o parent dirs or file extension

    for instrument in piano_rolls: # save each instrument's piano roll as a single image
        if debug: print("saving piano roll for ",instrument)
        piano_roll_to_img(piano_rolls[instrument], output_dir, midi_name, instrument, chords=chords, chord_names=chord_names, 
                          add_onsets=add_onsets, debug=debug)

    return

#### end of code copied from midi2img.py


#### below code originally in img2midi.py

def blockout_topbottom_arr(arr, border=CHORD_BORDER):
    "set the top and bottom border pixels to black"
    arr2 = arr.copy()
    arr2[:border, :] = 0
    arr2[-border:, :] = 0
    return arr2


def img2midi(img, draw_sep=512, debug=False):
    # operates on a single image
    # flip the image vertically because numpy and PIL have different ideas of what the first row is
    # if image vertical dimension is more than 128, then cut it into strips of 128 and concatenate them horizontally
    if debug: print(f"img2midi: img.size = {img.size}")
    if img.size[1] > 128:
        arr = np.concatenate([np.array(img.crop((0, i, img.size[0], i+128))) for i in range(0, img.size[1], 128)], axis=1)
    else:
        arr = np.array(img)
    if debug: print("0: arr.T[arr.T != 0] = ",arr.T[arr.T != 0])
    arr = blockout_topbottom_arr(arr)

    scale_factor = 0.5 # rgb down to velocity values
    piano_roll_array = np.array(arr*scale_factor, dtype=np.int32)
    piano_roll_array = np.flip(piano_roll_array, axis=0) # numpy as PIL are upside down relative to each other
    if debug: 
        print(f"piano_roll_array.shape = {piano_roll_array.shape}, piano_roll_array.dtype = {piano_roll_array.dtype}")
        print("1: piano_roll_array[piano_roll_array != 0] = ",piano_roll_array[piano_roll_array != 0])
    
    # draw a vertical line every 128/256/512 pixels
    if draw_sep > 0:
        line_every = draw_sep
        for i in range(0, piano_roll_array.shape[-1], line_every):
            if i>0: piano_roll_array[35:-35,i] = 30
    piano_roll_array = np.clip(piano_roll_array, 0, 127) # make sure velocities aren't out of bounds
    midi = piano_roll_to_pretty_midi(piano_roll_array)
    return midi


def flip_bottom_half_and_attach(sub_img):
    "takes one 256x256 and returns on 512x128 image with the bottom half reversed and attached on the right"
    h, w = sub_img.size
    new_img = Image.new(sub_img.mode, (w*2, h//2))
    new_img.paste(sub_img.crop((0, 0, w, h//2)), (0, 0))
    new_img.paste(sub_img.crop((0, h//2, w, h)).transpose(Image.FLIP_LEFT_RIGHT), (w, 0))
    return new_img 


def square_to_rect(img):
    #"""just an alias for flip_bottom_half_and_attach"""
    return flip_bottom_half_and_attach(img)

def rect_to_square(img):
    "takes a 512x128 image and returns a 256x256 image with the bottom half reversed"
    w, h = img.size
    new_img = Image.new(img.mode, (w//2, h*2))
    new_img.paste(img.crop((0, 0, w//2, h)), (0, 0))
    new_img.paste(img.crop((w//2, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT), (0, h))
    return new_img

def regroup_lines(img, debug=False): 
    """
    large images come in as an 8x8 grid of 256x256 images, in which the bottom half of each 256x256 is horizontally backwards
    we will rebuild this grid by first flipping the bottom half of each 256x256 image
    """
    img2 = Image.new('RGB', img.size)
    if debug: print(f"regroup_lines: img.size = {img.size}")

    if img.size[0] == 256:
        img2 = Image.new('RGB', (512,128))
    elif img.size[0] != 2048:
        if debug: print("regroup_lines: unexpected image size, returning image unchanged")
        return  img # no op, hope all's well
    imnum = 0
    for row in range(0, img.size[0], 256):
        for col in range(0, img.size[1], 256):
            imnum += 1
            sub_img = img.crop((col, row, col+256, row+256))
            sub_img = square_to_rect(sub_img)
            paste_x, paste_y = (imnum-1) % 4 * 512, (imnum-1) // 4 * 128
            if debug: print(f"imnum = {imnum}, paste_x = {paste_x}, paste_y = {paste_y}")
            img2.paste(sub_img, (paste_x, paste_y))
    if debug: img2.show()
    return img2


def is_red(r,g,b, thresh=20, debug=False):
    result = r > thresh and g < thresh and b < thresh
    if debug: print("is_red: r,g,b = ",r,g,b,", result = ",result)
    return result

def is_green(r,g,b, thresh=20):
    return r < thresh and g > thresh and b < thresh

def is_black(r,g,b, thresh=20):
    return r < thresh and g < thresh and b < thresh

def filter_redgreen(img:Image, 
                    require_onsets=True, # only keep green lines that start with a red pixel on the left
                    thresh=20, # minimum amount of red or green to count
                    onset_style=ONSET_STYLE, # 'start' or 'early'
                    debug=False):
    # filter: only keep black points, and green lines that start with a red pixel on the left.
    # i.e. only green points that have red or green to their left are valid notes
    # intended for img2midi 
    img.save('rgfilter_in.png')
    img2 = img.copy()
    if debug: print("img.size = ",img.size,", require_onsets = ",require_onsets," (not require_onsets) =",(not require_onsets)," thresh = ",thresh)
    w, h = img.size
    for y in range(CHORD_BORDER,h-CHORD_BORDER):
        note_on = False
        for x in range(w):  # scan from right to left
            r,g,b = img2.getpixel((x,y))               # pixel under consideration
            if debug and (r,g,b)!=(0,0,0): print(f"x, y: {x}, {y}:  r, g, b = {r},{g},{b}, note_on = {note_on}, is_red = {is_red(r,g,b, thresh)}, is_green = {is_green(r,g,b, thresh)}")
            if is_red(r,g,b, thresh): 
                note_on = True   
                if onset_style == 'start': # keep the note but change the red to green
                    img2.putpixel((x,y), (0,r,0))
            elif is_green(r,g,b, thresh) and require_onsets and note_on:
                img2.putpixel((x,y), (r,g,b)) # keep the note
            elif is_green(r,g,b, thresh) and (not require_onsets):
                img2.putpixel((x,y), (r,g,b)) # keep the note
                note_on = True 
            else:
                note_on = False
                img2.putpixel((x,y), (0,0,0))  # zero it out
    
    img2.save('rgfilter_out.png') # debugging always on here
    return img2

def arr_check(img, tag=''):
        img = img.convert("RGB")
        arr = np.array(img)[:,:,1]
        print(tag,": arr.shape = ",arr.shape, flush=True)
        print(tag,": arr.T[arr.T != 0] = ",arr.T[arr.T != 0], flush=True)

def img2midi_multi(img, require_onsets=True, separators=512, debug=False):
    "can operate on a grid of images"
    img = img.convert('RGB') 
    img = regroup_lines(img)
    img = filter_redgreen(img, require_onsets=require_onsets)
    #img = img.convert('L') # convert to grayscale
    red_arr = np.array(img.split()[0])
    green_arr = np.array(img.split()[1])
    combined_arr = red_arr + green_arr
    if debug: arr_check(img, '1')
    img = Image.fromarray(combined_arr, mode="L")
    if debug: arr_check(img, '2')
    return img2midi(img, draw_sep=separators)

def img_file_2_midi_file(img_file, output_dir='', require_onsets=True, separators=512, debug=False):
    "Converts an image file to a midi file"
    if debug: print(f"Processing {img_file}", flush=True)
    img = Image.open(img_file)
    if debug: arr_check(img, '0')
    midi = img2midi_multi(img, require_onsets=require_onsets, separators=separators)
    midi_file = os.path.basename(img_file).replace('.png', '.mid')
    if output_dir is not None and output_dir != '':
        midi_file = os.path.join(output_dir, midi_file)
    midi.write(midi_file)
    return midi_file



#### end of code copied from img2midi.py



### dataset routines, called from train.h

class RandomVerticalShift(torch.nn.Module):
    """
    Update: UNUSED.  Instead we do all transposing as pre-processing to facilitate chord detection.
    Randomly shift the image vertically by up to max_shift pixels, which correspond to semitones.
    """
    def __init__(self, max_shift=12):
        super().__init__()
        self.max_shift = max_shift

    def __call__(self, img):
        shift = torch.randint(-self.max_shift, self.max_shift, (1,))
        return self.vertical_shift(img, shift.item())

    def vertical_shift(self, img, shift):
        img = ImageOps.exif_transpose(img)  # Handle EXIF Orientation
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, shift), fillcolor=0)
        return img



class RandomBarCrop(torch.nn.Module):
    """
    Given a PIL image of a piano roll (non-square!), do random cropping the level of measures, i.e. bars, i.e. 16 16th-note pixels
    NOTE: might be nice if piano roll images have initial silence pre-removed -- assuming the first note is supposed to start on the first beat
    """
    def __init__(self, bar_length=16, window_length=512):
        super().__init__()
        self.bl = bar_length  # in 16th notes (16 pixels)
        self.wl = window_length  # in pixels
        self.bic = self.wl // self.bl  # bars in crop

    def __call__(self, img: Image, debug=False):
        bars_in_image = img.size[0] // self.bl  # number of bars in full image
        if self.bic >= bars_in_image: # pad horizontal end of image with zeros if needed
            pad = self.wl - img.size[0] + 1
            img = ImageOps.expand(img, (0, 0, pad, 0), fill=0)
            bars_in_image = img.size[0] // self.bl
        try:
            start_ind = torch.randint(0, bars_in_image - self.bic+1, (1,)).item() # start index of crop
        except Exception as e:
            print(f"***MY ERROR: {e}.  bars_in_image = {bars_in_image}, self.bic = {self.bic}")
            assert False
        start_pixel = start_ind * self.bl  # start pixel of crop
        new_img = img.crop((start_pixel, 0, start_pixel + self.wl, img.size[1]))
        assert new_img.size[0] == self.wl and new_img.size[1]==128, f"ERROR: new_img.size = {new_img.size}, self.wl = {self.wl}"
        return new_img



class StackPianoRollsImage(torch.nn.Module):
    """
    Given a PIL image of a piano roll, cut in in half horizontally, 
    stack the two halves, with the lower half mirrored horzontally.
    """
    def __init__(self, final_size=(256, 256), max_shift=13):
        super().__init__()
        self.final_size = final_size

    def __call__(self, img: Image, debug=False):
        if img.size[0] <= 128 and img.size[1] <= 128:
            return img # don't stack small images
        # image dimensions are likely 512x128. I want 256x256 output
        half_width = img.size[0] // 2
        #make a new image with dimensions 256x256, with the same color mode as img
        new_img = Image.new(img.mode, self.final_size)
        # paste the first half of the image into the top half of the new image
        first_half = img.crop((0, 0, half_width, img.size[1]))
        new_img.paste(first_half, (0, 0))
        # paste the second half of the image into the bottom half of the new image, but flipped horizontally
        next_half = img.crop((half_width, 0, 2*half_width, img.size[1])) 
        next_half = ImageOps.mirror(next_half)
        new_img.paste(next_half, (0, img.size[1]))
        return new_img



class StackPianoRollsTensor(torch.nn.Module):   
    """
    Tensor version of StackPianoRollsImage.  Unused, i think. 
    Given a torch tensor of a piano roll, cut in in half horizontally, stack the two halves
    but have the bottom half mirrored horzontally.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor):
        if img.shape[0] <= 128 and img.shape[1] <= 128:
            return img # don't stack small images
        img = img.permute(1, 2, 0)
        half_width = img.shape[0] // 2
        img = torch.cat([img[:half_width], img[half_width:][::-1]], dim=0)
        img = img.permute(2, 0, 1)
        return img
    



if __name__ == '__main__':
    import sys 

    # testing for  the StackPianoRollsImage class
    filename = sys.argv[-1]
    print("filename = ", filename)
    img = Image.open(filename)
    img = transforms.RandomCrop((128, 512))(img) # randomly crop it to 128x512
    img = StackPianoRollsImage()(img, debug=True)
    img.show()

