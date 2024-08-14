# Video_processor
BIT LAB 810: Ice skating project

Run code in the interpreter 

or in the command line:

python .\refactored_video_processor.py --file_type {video type | Default: .mov} --input_path {path | Default: '.'} --output_folder {path | Default: '.'}


Current approach:

Process the video in batches of 5 frames (This can be increased if your PC can handle it). The pose detection is done on every frame and also overlayed on the video every frame. The frames are NOT immediately committed to the drive to decrease I/O operations. This does come at the cost of higher RAM used, but I think it's worth it. I'll keep working on this when I have some more time.
