# MIT License
#
# Copyright (c) 2018, Andrew Warrington
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
util.py
AW

TL;DR -- A number of useful utility functions to be used.
'''


def echo_to_file(_f, _str, _dont_display=False):
	'''
	AW - echo_to_file - instead of using `print', use this function, that
	wraps the call to print with a file write, for writing reports that are
	accessible during execution. If the file does not exist, the file is created.
	:param _f: string containing the relative or absolute file path to the file.
	:param _str: the string to be written to the file.
	:param _dont_display: do not display to stdout, just echo to file.
	:return: None
	'''
	with open(_f, mode='a+') as fid:
		fid.write(_str + '\n')
	if not _dont_display:
		print(_str)

