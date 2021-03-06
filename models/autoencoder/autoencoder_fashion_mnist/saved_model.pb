¥:
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??3
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: *
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
: *
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
: *
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_24/gamma
?
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_24/beta
?
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
:@*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_25/gamma
?
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_25/beta
?
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes	
:?*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
??*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:?*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	?*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? * 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	? *
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:? *
dtype0
?
conv2d_transpose_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameconv2d_transpose_16/kernel
?
.conv2d_transpose_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameconv2d_transpose_16/bias
?
,conv2d_transpose_16/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_26/gamma
?
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_26/beta
?
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_17/kernel
?
.conv2d_transpose_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_transpose_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_17/bias
?
,conv2d_transpose_17/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:@*
dtype0
?
conv2d_transpose_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_18/kernel
?
.conv2d_transpose_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_18/bias
?
,conv2d_transpose_18/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/bias*
_output_shapes
: *
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
: *
dtype0
?
conv2d_transpose_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_19/kernel
?
.conv2d_transpose_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_19/bias
?
,conv2d_transpose_19/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/bias*
_output_shapes
:*
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_24/moving_mean
?
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_24/moving_variance
?
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_25/moving_mean
?
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_25/moving_variance
?
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_26/moving_mean
?
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_26/moving_variance
?
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_11/kernel/m
?
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_23/gamma/m
?
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_23/beta/m
?
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_12/kernel/m
?
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_24/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_24/gamma/m
?
7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_24/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_24/beta/m
?
6Adam/batch_normalization_24/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_13/kernel/m
?
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_13/bias/m
|
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_25/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_25/gamma/m
?
7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_25/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_25/beta/m
?
6Adam/batch_normalization_25/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes
:	? *
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_nameAdam/dense_14/bias/m
z
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes	
:? *
dtype0
?
!Adam/conv2d_transpose_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*2
shared_name#!Adam/conv2d_transpose_16/kernel/m
?
5Adam/conv2d_transpose_16/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_16/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_transpose_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/conv2d_transpose_16/bias/m
?
3Adam/conv2d_transpose_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_16/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_26/gamma/m
?
7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_26/beta/m
?
6Adam/batch_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/m*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*2
shared_name#!Adam/conv2d_transpose_17/kernel/m
?
5Adam/conv2d_transpose_17/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_17/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_transpose_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_17/bias/m
?
3Adam/conv2d_transpose_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_17/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_27/gamma/m
?
7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_27/beta/m
?
6Adam/batch_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_18/kernel/m
?
5Adam/conv2d_transpose_18/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_18/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_18/bias/m
?
3Adam/conv2d_transpose_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_18/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_28/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_28/gamma/m
?
7Adam/batch_normalization_28/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_28/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_28/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_28/beta/m
?
6Adam/batch_normalization_28/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_28/beta/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_19/kernel/m
?
5Adam/conv2d_transpose_19/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_19/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_19/bias/m
?
3Adam/conv2d_transpose_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_19/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_11/kernel/v
?
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_23/gamma/v
?
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_23/beta/v
?
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_12/kernel/v
?
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_24/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_24/gamma/v
?
7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_24/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_24/beta/v
?
6Adam/batch_normalization_24/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_13/kernel/v
?
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_13/bias/v
|
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_25/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_25/gamma/v
?
7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_25/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_25/beta/v
?
6Adam/batch_normalization_25/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes
:	? *
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_nameAdam/dense_14/bias/v
z
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes	
:? *
dtype0
?
!Adam/conv2d_transpose_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*2
shared_name#!Adam/conv2d_transpose_16/kernel/v
?
5Adam/conv2d_transpose_16/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_16/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_transpose_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/conv2d_transpose_16/bias/v
?
3Adam/conv2d_transpose_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_16/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_26/gamma/v
?
7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_26/beta/v
?
6Adam/batch_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/v*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*2
shared_name#!Adam/conv2d_transpose_17/kernel/v
?
5Adam/conv2d_transpose_17/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_17/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_transpose_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_17/bias/v
?
3Adam/conv2d_transpose_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_17/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_27/gamma/v
?
7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_27/beta/v
?
6Adam/batch_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_18/kernel/v
?
5Adam/conv2d_transpose_18/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_18/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_18/bias/v
?
3Adam/conv2d_transpose_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_18/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_28/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_28/gamma/v
?
7Adam/batch_normalization_28/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_28/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_28/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_28/beta/v
?
6Adam/batch_normalization_28/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_28/beta/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_19/kernel/v
?
5Adam/conv2d_transpose_19/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_19/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_19/bias/v
?
3Adam/conv2d_transpose_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_19/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
encoder
decoder
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
trainable_variables
	variables
regularization_losses
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
 layer_with_weights-3
 layer-6
!layer_with_weights-4
!layer-7
"layer-8
#layer_with_weights-5
#layer-9
$layer_with_weights-6
$layer-10
%layer-11
&layer_with_weights-7
&layer-12
'layer-13
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?
,iter

-beta_1

.beta_2
	/decay
0learning_rate1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31
?
10
21
32
43
Q4
R5
56
67
78
89
S10
T11
912
:13
;14
<15
U16
V17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
W28
X29
G30
H31
I32
J33
Y34
Z35
K36
L37
M38
N39
[40
\41
O42
P43
 
?
trainable_variables

]layers
^non_trainable_variables
	variables
regularization_losses
_metrics
`layer_metrics
alayer_regularization_losses
 
h

1kernel
2bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
?
faxis
	3gamma
4beta
Qmoving_mean
Rmoving_variance
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
R
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
h

5kernel
6bias
otrainable_variables
p	variables
qregularization_losses
r	keras_api
?
saxis
	7gamma
8beta
Smoving_mean
Tmoving_variance
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
R
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
h

9kernel
:bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
?
	?axis
	;gamma
<beta
Umoving_mean
Vmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

?kernel
@bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
v
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
?
10
21
32
43
Q4
R5
56
67
78
89
S10
T11
912
:13
;14
<15
U16
V17
=18
>19
?20
@21
 
?
trainable_variables
?layers
?non_trainable_variables
	variables
regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
l

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	Egamma
Fbeta
Wmoving_mean
Xmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Gkernel
Hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	Igamma
Jbeta
Ymoving_mean
Zmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Kkernel
Lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	Mgamma
Nbeta
[moving_mean
\moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
v
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
?
A0
B1
C2
D3
E4
F5
W6
X7
G8
H9
I10
J11
Y12
Z13
K14
L15
M16
N17
[18
\19
O20
P21
 
?
(trainable_variables
?layers
?non_trainable_variables
)	variables
*regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_11/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_11/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_23/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_23/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_12/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_12/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_24/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_24/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_13/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_13/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_25/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_25/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_12/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_12/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_13/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_13/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_14/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_14/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_16/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_16/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_26/gamma1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_26/beta1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_17/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_17/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_27/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_27/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_18/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_18/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_28/gamma1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_28/beta1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_19/kernel1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_19/bias1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_23/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_23/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_24/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_24/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_25/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_25/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_26/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_26/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_27/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_27/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_28/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_28/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE

0
1
V
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11

?0
 
 

10
21

10
21
 
?
btrainable_variables
?layers
?non_trainable_variables
c	variables
dregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

30
41

30
41
Q2
R3
 
?
gtrainable_variables
?layers
?non_trainable_variables
h	variables
iregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
ktrainable_variables
?layers
?non_trainable_variables
l	variables
mregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

50
61

50
61
 
?
otrainable_variables
?layers
?non_trainable_variables
p	variables
qregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

70
81

70
81
S2
T3
 
?
ttrainable_variables
?layers
?non_trainable_variables
u	variables
vregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
xtrainable_variables
?layers
?non_trainable_variables
y	variables
zregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

90
:1

90
:1
 
?
|trainable_variables
?layers
?non_trainable_variables
}	variables
~regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

;0
<1

;0
<1
U2
V3
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

=0
>1

=0
>1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

?0
@1

?0
@1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
*
Q0
R1
S2
T3
U4
V5
 
 
 

A0
B1

A0
B1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

C0
D1

C0
D1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

E0
F1

E0
F1
W2
X3
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

G0
H1

G0
H1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

I0
J1

I0
J1
Y2
Z3
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

K0
L1

K0
L1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

M0
N1

M0
N1
[2
\3
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

O0
P1

O0
P1
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
f
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
*
W0
X1
Y2
Z3
[4
\5
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 

Q0
R1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

S0
T1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

U0
V1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Y0
Z1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

[0
\1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_12/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_12/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_24/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_13/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_13/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_25/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_12/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_12/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_13/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_13/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_14/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_14/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_16/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_16/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_26/beta/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_17/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_17/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_27/beta/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_18/kernel/mMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_18/bias/mMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_28/gamma/mMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_28/beta/mMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_19/kernel/mMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_19/bias/mMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_12/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_12/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_24/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_13/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_13/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_25/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_12/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_12/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_13/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_13/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_14/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_14/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_16/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_16/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_26/beta/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_17/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_17/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_27/beta/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_18/kernel/vMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_18/bias/vMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_28/gamma/vMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_28/beta/vMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_19/kernel/vMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_19/bias/vMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_11/kernelconv2d_11/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_variancedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_transpose_17/kernelconv2d_transpose_17/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_transpose_18/kernelconv2d_transpose_18/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_19/kernelconv2d_transpose_19/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_311141
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp.conv2d_transpose_16/kernel/Read/ReadVariableOp,conv2d_transpose_16/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp.conv2d_transpose_17/kernel/Read/ReadVariableOp,conv2d_transpose_17/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp.conv2d_transpose_18/kernel/Read/ReadVariableOp,conv2d_transpose_18/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp.conv2d_transpose_19/kernel/Read/ReadVariableOp,conv2d_transpose_19/bias/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_24/beta/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_25/beta/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_16/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_16/bias/m/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_26/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_17/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_27/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_18/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_28/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_28/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_19/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_19/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_24/beta/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_25/beta/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_16/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_16/bias/v/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_26/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_17/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_27/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_18/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_28/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_28/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_19/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_19/bias/v/Read/ReadVariableOpConst*?
Tiny
w2u	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_314647
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_11/kernelconv2d_11/biasbatch_normalization_23/gammabatch_normalization_23/betaconv2d_12/kernelconv2d_12/biasbatch_normalization_24/gammabatch_normalization_24/betaconv2d_13/kernelconv2d_13/biasbatch_normalization_25/gammabatch_normalization_25/betadense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasbatch_normalization_26/gammabatch_normalization_26/betaconv2d_transpose_17/kernelconv2d_transpose_17/biasbatch_normalization_27/gammabatch_normalization_27/betaconv2d_transpose_18/kernelconv2d_transpose_18/biasbatch_normalization_28/gammabatch_normalization_28/betaconv2d_transpose_19/kernelconv2d_transpose_19/bias"batch_normalization_23/moving_mean&batch_normalization_23/moving_variance"batch_normalization_24/moving_mean&batch_normalization_24/moving_variance"batch_normalization_25/moving_mean&batch_normalization_25/moving_variance"batch_normalization_26/moving_mean&batch_normalization_26/moving_variance"batch_normalization_27/moving_mean&batch_normalization_27/moving_variance"batch_normalization_28/moving_mean&batch_normalization_28/moving_variancetotalcountAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/m#Adam/batch_normalization_24/gamma/m"Adam/batch_normalization_24/beta/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/m#Adam/batch_normalization_25/gamma/m"Adam/batch_normalization_25/beta/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/m!Adam/conv2d_transpose_16/kernel/mAdam/conv2d_transpose_16/bias/m#Adam/batch_normalization_26/gamma/m"Adam/batch_normalization_26/beta/m!Adam/conv2d_transpose_17/kernel/mAdam/conv2d_transpose_17/bias/m#Adam/batch_normalization_27/gamma/m"Adam/batch_normalization_27/beta/m!Adam/conv2d_transpose_18/kernel/mAdam/conv2d_transpose_18/bias/m#Adam/batch_normalization_28/gamma/m"Adam/batch_normalization_28/beta/m!Adam/conv2d_transpose_19/kernel/mAdam/conv2d_transpose_19/bias/mAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/v#Adam/batch_normalization_24/gamma/v"Adam/batch_normalization_24/beta/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/v#Adam/batch_normalization_25/gamma/v"Adam/batch_normalization_25/beta/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v!Adam/conv2d_transpose_16/kernel/vAdam/conv2d_transpose_16/bias/v#Adam/batch_normalization_26/gamma/v"Adam/batch_normalization_26/beta/v!Adam/conv2d_transpose_17/kernel/vAdam/conv2d_transpose_17/bias/v#Adam/batch_normalization_27/gamma/v"Adam/batch_normalization_27/beta/v!Adam/conv2d_transpose_18/kernel/vAdam/conv2d_transpose_18/bias/v#Adam/batch_normalization_28/gamma/v"Adam/batch_normalization_28/beta/v!Adam/conv2d_transpose_19/kernel/vAdam/conv2d_transpose_19/bias/v*
Tinx
v2t*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_315002??.
?
?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
D__inference_dense_13_layer_call_and_return_conditional_losses_309053

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_309637

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?J
?	
H__inference_sequential_9_layer_call_and_return_conditional_losses_310214

inputs
dense_14_310155
dense_14_310157
conv2d_transpose_16_310162
conv2d_transpose_16_310164!
batch_normalization_26_310167!
batch_normalization_26_310169!
batch_normalization_26_310171!
batch_normalization_26_310173
conv2d_transpose_17_310177
conv2d_transpose_17_310179!
batch_normalization_27_310182!
batch_normalization_27_310184!
batch_normalization_27_310186!
batch_normalization_27_310188
conv2d_transpose_18_310192
conv2d_transpose_18_310194!
batch_normalization_28_310197!
batch_normalization_28_310199!
batch_normalization_28_310201!
batch_normalization_28_310203
conv2d_transpose_19_310207
conv2d_transpose_19_310209
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_310155dense_14_310157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_3098582"
 dense_14/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_39_layer_call_and_return_conditional_losses_3098792
activation_39/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_3099012
reshape_4/PartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_16_310162conv2d_transpose_16_310164*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_3093862-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_26_310167batch_normalization_26_310169batch_normalization_26_310171batch_normalization_26_310173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_30945820
.batch_normalization_26/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_40_layer_call_and_return_conditional_losses_3099542
activation_40/PartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall&activation_40/PartitionedCall:output:0conv2d_transpose_17_310177conv2d_transpose_17_310179*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_3095342-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_27_310182batch_normalization_27_310184batch_normalization_27_310186batch_normalization_27_310188*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_30960620
.batch_normalization_27/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_41_layer_call_and_return_conditional_losses_3100072
activation_41/PartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall&activation_41/PartitionedCall:output:0conv2d_transpose_18_310192conv2d_transpose_18_310194*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_3096822-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_28_310197batch_normalization_28_310199batch_normalization_28_310201batch_normalization_28_310203*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_30975420
.batch_normalization_28/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_42_layer_call_and_return_conditional_losses_3100602
activation_42/PartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_transpose_19_310207conv2d_transpose_19_310209*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_3098342-
+conv2d_transpose_19/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_43_layer_call_and_return_conditional_losses_3100782
activation_43/PartitionedCall?
IdentityIdentity&activation_43/PartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313508

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_314009

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?C
"__inference__traced_restore_315002
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate'
#assignvariableop_5_conv2d_11_kernel%
!assignvariableop_6_conv2d_11_bias3
/assignvariableop_7_batch_normalization_23_gamma2
.assignvariableop_8_batch_normalization_23_beta'
#assignvariableop_9_conv2d_12_kernel&
"assignvariableop_10_conv2d_12_bias4
0assignvariableop_11_batch_normalization_24_gamma3
/assignvariableop_12_batch_normalization_24_beta(
$assignvariableop_13_conv2d_13_kernel&
"assignvariableop_14_conv2d_13_bias4
0assignvariableop_15_batch_normalization_25_gamma3
/assignvariableop_16_batch_normalization_25_beta'
#assignvariableop_17_dense_12_kernel%
!assignvariableop_18_dense_12_bias'
#assignvariableop_19_dense_13_kernel%
!assignvariableop_20_dense_13_bias'
#assignvariableop_21_dense_14_kernel%
!assignvariableop_22_dense_14_bias2
.assignvariableop_23_conv2d_transpose_16_kernel0
,assignvariableop_24_conv2d_transpose_16_bias4
0assignvariableop_25_batch_normalization_26_gamma3
/assignvariableop_26_batch_normalization_26_beta2
.assignvariableop_27_conv2d_transpose_17_kernel0
,assignvariableop_28_conv2d_transpose_17_bias4
0assignvariableop_29_batch_normalization_27_gamma3
/assignvariableop_30_batch_normalization_27_beta2
.assignvariableop_31_conv2d_transpose_18_kernel0
,assignvariableop_32_conv2d_transpose_18_bias4
0assignvariableop_33_batch_normalization_28_gamma3
/assignvariableop_34_batch_normalization_28_beta2
.assignvariableop_35_conv2d_transpose_19_kernel0
,assignvariableop_36_conv2d_transpose_19_bias:
6assignvariableop_37_batch_normalization_23_moving_mean>
:assignvariableop_38_batch_normalization_23_moving_variance:
6assignvariableop_39_batch_normalization_24_moving_mean>
:assignvariableop_40_batch_normalization_24_moving_variance:
6assignvariableop_41_batch_normalization_25_moving_mean>
:assignvariableop_42_batch_normalization_25_moving_variance:
6assignvariableop_43_batch_normalization_26_moving_mean>
:assignvariableop_44_batch_normalization_26_moving_variance:
6assignvariableop_45_batch_normalization_27_moving_mean>
:assignvariableop_46_batch_normalization_27_moving_variance:
6assignvariableop_47_batch_normalization_28_moving_mean>
:assignvariableop_48_batch_normalization_28_moving_variance
assignvariableop_49_total
assignvariableop_50_count/
+assignvariableop_51_adam_conv2d_11_kernel_m-
)assignvariableop_52_adam_conv2d_11_bias_m;
7assignvariableop_53_adam_batch_normalization_23_gamma_m:
6assignvariableop_54_adam_batch_normalization_23_beta_m/
+assignvariableop_55_adam_conv2d_12_kernel_m-
)assignvariableop_56_adam_conv2d_12_bias_m;
7assignvariableop_57_adam_batch_normalization_24_gamma_m:
6assignvariableop_58_adam_batch_normalization_24_beta_m/
+assignvariableop_59_adam_conv2d_13_kernel_m-
)assignvariableop_60_adam_conv2d_13_bias_m;
7assignvariableop_61_adam_batch_normalization_25_gamma_m:
6assignvariableop_62_adam_batch_normalization_25_beta_m.
*assignvariableop_63_adam_dense_12_kernel_m,
(assignvariableop_64_adam_dense_12_bias_m.
*assignvariableop_65_adam_dense_13_kernel_m,
(assignvariableop_66_adam_dense_13_bias_m.
*assignvariableop_67_adam_dense_14_kernel_m,
(assignvariableop_68_adam_dense_14_bias_m9
5assignvariableop_69_adam_conv2d_transpose_16_kernel_m7
3assignvariableop_70_adam_conv2d_transpose_16_bias_m;
7assignvariableop_71_adam_batch_normalization_26_gamma_m:
6assignvariableop_72_adam_batch_normalization_26_beta_m9
5assignvariableop_73_adam_conv2d_transpose_17_kernel_m7
3assignvariableop_74_adam_conv2d_transpose_17_bias_m;
7assignvariableop_75_adam_batch_normalization_27_gamma_m:
6assignvariableop_76_adam_batch_normalization_27_beta_m9
5assignvariableop_77_adam_conv2d_transpose_18_kernel_m7
3assignvariableop_78_adam_conv2d_transpose_18_bias_m;
7assignvariableop_79_adam_batch_normalization_28_gamma_m:
6assignvariableop_80_adam_batch_normalization_28_beta_m9
5assignvariableop_81_adam_conv2d_transpose_19_kernel_m7
3assignvariableop_82_adam_conv2d_transpose_19_bias_m/
+assignvariableop_83_adam_conv2d_11_kernel_v-
)assignvariableop_84_adam_conv2d_11_bias_v;
7assignvariableop_85_adam_batch_normalization_23_gamma_v:
6assignvariableop_86_adam_batch_normalization_23_beta_v/
+assignvariableop_87_adam_conv2d_12_kernel_v-
)assignvariableop_88_adam_conv2d_12_bias_v;
7assignvariableop_89_adam_batch_normalization_24_gamma_v:
6assignvariableop_90_adam_batch_normalization_24_beta_v/
+assignvariableop_91_adam_conv2d_13_kernel_v-
)assignvariableop_92_adam_conv2d_13_bias_v;
7assignvariableop_93_adam_batch_normalization_25_gamma_v:
6assignvariableop_94_adam_batch_normalization_25_beta_v.
*assignvariableop_95_adam_dense_12_kernel_v,
(assignvariableop_96_adam_dense_12_bias_v.
*assignvariableop_97_adam_dense_13_kernel_v,
(assignvariableop_98_adam_dense_13_bias_v.
*assignvariableop_99_adam_dense_14_kernel_v-
)assignvariableop_100_adam_dense_14_bias_v:
6assignvariableop_101_adam_conv2d_transpose_16_kernel_v8
4assignvariableop_102_adam_conv2d_transpose_16_bias_v<
8assignvariableop_103_adam_batch_normalization_26_gamma_v;
7assignvariableop_104_adam_batch_normalization_26_beta_v:
6assignvariableop_105_adam_conv2d_transpose_17_kernel_v8
4assignvariableop_106_adam_conv2d_transpose_17_bias_v<
8assignvariableop_107_adam_batch_normalization_27_gamma_v;
7assignvariableop_108_adam_batch_normalization_27_beta_v:
6assignvariableop_109_adam_conv2d_transpose_18_kernel_v8
4assignvariableop_110_adam_conv2d_transpose_18_bias_v<
8assignvariableop_111_adam_batch_normalization_28_gamma_v;
7assignvariableop_112_adam_batch_normalization_28_beta_v:
6assignvariableop_113_adam_conv2d_transpose_19_kernel_v8
4assignvariableop_114_adam_conv2d_transpose_19_bias_v
identity_116??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?:
value?:B?:tB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypesx
v2t	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_11_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_11_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_23_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_23_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_12_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_12_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_24_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_24_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_13_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_13_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_25_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_25_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_12_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_12_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_13_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_13_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_14_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_14_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_conv2d_transpose_16_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_conv2d_transpose_16_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_batch_normalization_26_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_26_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_conv2d_transpose_17_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_conv2d_transpose_17_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_batch_normalization_27_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_27_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_conv2d_transpose_18_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_conv2d_transpose_18_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_batch_normalization_28_gammaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp/assignvariableop_34_batch_normalization_28_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_conv2d_transpose_19_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_conv2d_transpose_19_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp6assignvariableop_37_batch_normalization_23_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp:assignvariableop_38_batch_normalization_23_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_batch_normalization_24_moving_meanIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp:assignvariableop_40_batch_normalization_24_moving_varianceIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_batch_normalization_25_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp:assignvariableop_42_batch_normalization_25_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_batch_normalization_26_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp:assignvariableop_44_batch_normalization_26_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_batch_normalization_27_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp:assignvariableop_46_batch_normalization_27_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_batch_normalization_28_moving_meanIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp:assignvariableop_48_batch_normalization_28_moving_varianceIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_11_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_11_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_23_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_23_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_12_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_12_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_24_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_24_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_13_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_13_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_batch_normalization_25_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_batch_normalization_25_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_12_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_12_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_13_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_13_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_14_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_14_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_conv2d_transpose_16_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp3assignvariableop_70_adam_conv2d_transpose_16_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_26_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_26_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_conv2d_transpose_17_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_conv2d_transpose_17_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_27_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_27_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_conv2d_transpose_18_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adam_conv2d_transpose_18_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_28_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_28_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp5assignvariableop_81_adam_conv2d_transpose_19_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adam_conv2d_transpose_19_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_11_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_11_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_batch_normalization_23_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_batch_normalization_23_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_conv2d_12_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_conv2d_12_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_24_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_24_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv2d_13_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv2d_13_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_batch_normalization_25_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_batch_normalization_25_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_dense_12_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_dense_12_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_13_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_13_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_14_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_14_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp6assignvariableop_101_adam_conv2d_transpose_16_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp4assignvariableop_102_adam_conv2d_transpose_16_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp8assignvariableop_103_adam_batch_normalization_26_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_26_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_conv2d_transpose_17_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp4assignvariableop_106_adam_conv2d_transpose_17_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_batch_normalization_27_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_27_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp6assignvariableop_109_adam_conv2d_transpose_18_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp4assignvariableop_110_adam_conv2d_transpose_18_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_28_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_28_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp6assignvariableop_113_adam_conv2d_transpose_19_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp4assignvariableop_114_adam_conv2d_transpose_19_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_115Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_115?
Identity_116IdentityIdentity_115:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_116"%
identity_116Identity_116:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?2
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312014
x9
5sequential_8_conv2d_11_conv2d_readvariableop_resource:
6sequential_8_conv2d_11_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_23_readvariableop_resourceA
=sequential_8_batch_normalization_23_readvariableop_1_resourceP
Lsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_12_conv2d_readvariableop_resource:
6sequential_8_conv2d_12_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_24_readvariableop_resourceA
=sequential_8_batch_normalization_24_readvariableop_1_resourceP
Lsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_13_conv2d_readvariableop_resource:
6sequential_8_conv2d_13_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_25_readvariableop_resourceA
=sequential_8_batch_normalization_25_readvariableop_1_resourceP
Lsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8
4sequential_8_dense_12_matmul_readvariableop_resource9
5sequential_8_dense_12_biasadd_readvariableop_resource8
4sequential_8_dense_13_matmul_readvariableop_resource9
5sequential_8_dense_13_biasadd_readvariableop_resource8
4sequential_9_dense_14_matmul_readvariableop_resource9
5sequential_9_dense_14_biasadd_readvariableop_resourceM
Isequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_26_readvariableop_resourceA
=sequential_9_batch_normalization_26_readvariableop_1_resourceP
Lsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_27_readvariableop_resourceA
=sequential_9_batch_normalization_27_readvariableop_1_resourceP
Lsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_28_readvariableop_resourceA
=sequential_9_batch_normalization_28_readvariableop_1_resourceP
Lsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource
identity??2sequential_8/batch_normalization_23/AssignNewValue?4sequential_8/batch_normalization_23/AssignNewValue_1?Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_23/ReadVariableOp?4sequential_8/batch_normalization_23/ReadVariableOp_1?2sequential_8/batch_normalization_24/AssignNewValue?4sequential_8/batch_normalization_24/AssignNewValue_1?Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_24/ReadVariableOp?4sequential_8/batch_normalization_24/ReadVariableOp_1?2sequential_8/batch_normalization_25/AssignNewValue?4sequential_8/batch_normalization_25/AssignNewValue_1?Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_25/ReadVariableOp?4sequential_8/batch_normalization_25/ReadVariableOp_1?-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?,sequential_8/conv2d_11/Conv2D/ReadVariableOp?-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?,sequential_8/conv2d_12/Conv2D/ReadVariableOp?-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?,sequential_8/conv2d_13/Conv2D/ReadVariableOp?,sequential_8/dense_12/BiasAdd/ReadVariableOp?+sequential_8/dense_12/MatMul/ReadVariableOp?,sequential_8/dense_13/BiasAdd/ReadVariableOp?+sequential_8/dense_13/MatMul/ReadVariableOp?2sequential_9/batch_normalization_26/AssignNewValue?4sequential_9/batch_normalization_26/AssignNewValue_1?Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_26/ReadVariableOp?4sequential_9/batch_normalization_26/ReadVariableOp_1?2sequential_9/batch_normalization_27/AssignNewValue?4sequential_9/batch_normalization_27/AssignNewValue_1?Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_27/ReadVariableOp?4sequential_9/batch_normalization_27/ReadVariableOp_1?2sequential_9/batch_normalization_28/AssignNewValue?4sequential_9/batch_normalization_28/AssignNewValue_1?Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_28/ReadVariableOp?4sequential_9/batch_normalization_28/ReadVariableOp_1?7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?,sequential_9/dense_14/BiasAdd/ReadVariableOp?+sequential_9/dense_14/MatMul/ReadVariableOp?
,sequential_8/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_11/Conv2D/ReadVariableOp?
sequential_8/conv2d_11/Conv2DConv2Dx4sequential_8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_8/conv2d_11/Conv2D?
-sequential_8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?
sequential_8/conv2d_11/BiasAddBiasAdd&sequential_8/conv2d_11/Conv2D:output:05sequential_8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2 
sequential_8/conv2d_11/BiasAdd?
2sequential_8/batch_normalization_23/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_8/batch_normalization_23/ReadVariableOp?
4sequential_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_8/batch_normalization_23/ReadVariableOp_1?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_11/BiasAdd:output:0:sequential_8/batch_normalization_23/ReadVariableOp:value:0<sequential_8/batch_normalization_23/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_23/FusedBatchNormV3?
2sequential_8/batch_normalization_23/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_23/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_23/AssignNewValue?
4sequential_8/batch_normalization_23/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_23/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_23/AssignNewValue_1?
sequential_8/activation_35/ReluRelu8sequential_8/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2!
sequential_8/activation_35/Relu?
,sequential_8/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_8/conv2d_12/Conv2D/ReadVariableOp?
sequential_8/conv2d_12/Conv2DConv2D-sequential_8/activation_35/Relu:activations:04sequential_8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_8/conv2d_12/Conv2D?
-sequential_8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?
sequential_8/conv2d_12/BiasAddBiasAdd&sequential_8/conv2d_12/Conv2D:output:05sequential_8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_8/conv2d_12/BiasAdd?
2sequential_8/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_8/batch_normalization_24/ReadVariableOp?
4sequential_8/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_8/batch_normalization_24/ReadVariableOp_1?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_12/BiasAdd:output:0:sequential_8/batch_normalization_24/ReadVariableOp:value:0<sequential_8/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_24/FusedBatchNormV3?
2sequential_8/batch_normalization_24/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_24/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_24/AssignNewValue?
4sequential_8/batch_normalization_24/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_24/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_24/AssignNewValue_1?
sequential_8/activation_36/ReluRelu8sequential_8/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_8/activation_36/Relu?
,sequential_8/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02.
,sequential_8/conv2d_13/Conv2D/ReadVariableOp?
sequential_8/conv2d_13/Conv2DConv2D-sequential_8/activation_36/Relu:activations:04sequential_8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
sequential_8/conv2d_13/Conv2D?
-sequential_8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?
sequential_8/conv2d_13/BiasAddBiasAdd&sequential_8/conv2d_13/Conv2D:output:05sequential_8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_13/BiasAdd?
2sequential_8/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_25/ReadVariableOp?
4sequential_8/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_25/ReadVariableOp_1?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_13/BiasAdd:output:0:sequential_8/batch_normalization_25/ReadVariableOp:value:0<sequential_8/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_25/FusedBatchNormV3?
2sequential_8/batch_normalization_25/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_25/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_25/AssignNewValue?
4sequential_8/batch_normalization_25/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_25/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_25/AssignNewValue_1?
sequential_8/activation_37/ReluRelu8sequential_8/batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_37/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape-sequential_8/activation_37/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
+sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_8/dense_12/MatMul/ReadVariableOp?
sequential_8/dense_12/MatMulMatMul'sequential_8/flatten_4/Reshape:output:03sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/MatMul?
,sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_8/dense_12/BiasAdd/ReadVariableOp?
sequential_8/dense_12/BiasAddBiasAdd&sequential_8/dense_12/MatMul:product:04sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/BiasAdd?
sequential_8/activation_38/ReluRelu&sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_8/activation_38/Relu?
+sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_8/dense_13/MatMul/ReadVariableOp?
sequential_8/dense_13/MatMulMatMul-sequential_8/activation_38/Relu:activations:03sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/MatMul?
,sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_8/dense_13/BiasAdd/ReadVariableOp?
sequential_8/dense_13/BiasAddBiasAdd&sequential_8/dense_13/MatMul:product:04sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/BiasAdd?
+sequential_9/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02-
+sequential_9/dense_14/MatMul/ReadVariableOp?
sequential_9/dense_14/MatMulMatMul&sequential_8/dense_13/BiasAdd:output:03sequential_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/MatMul?
,sequential_9/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02.
,sequential_9/dense_14/BiasAdd/ReadVariableOp?
sequential_9/dense_14/BiasAddBiasAdd&sequential_9/dense_14/MatMul:product:04sequential_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/BiasAdd?
sequential_9/activation_39/ReluRelu&sequential_9/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2!
sequential_9/activation_39/Relu?
sequential_9/reshape_4/ShapeShape-sequential_9/activation_39/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_4/Shape?
*sequential_9/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/reshape_4/strided_slice/stack?
,sequential_9/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_1?
,sequential_9/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_2?
$sequential_9/reshape_4/strided_sliceStridedSlice%sequential_9/reshape_4/Shape:output:03sequential_9/reshape_4/strided_slice/stack:output:05sequential_9/reshape_4/strided_slice/stack_1:output:05sequential_9/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/reshape_4/strided_slice?
&sequential_9/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/1?
&sequential_9/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/2?
&sequential_9/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_9/reshape_4/Reshape/shape/3?
$sequential_9/reshape_4/Reshape/shapePack-sequential_9/reshape_4/strided_slice:output:0/sequential_9/reshape_4/Reshape/shape/1:output:0/sequential_9/reshape_4/Reshape/shape/2:output:0/sequential_9/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/reshape_4/Reshape/shape?
sequential_9/reshape_4/ReshapeReshape-sequential_9/activation_39/Relu:activations:0-sequential_9/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2 
sequential_9/reshape_4/Reshape?
&sequential_9/conv2d_transpose_16/ShapeShape'sequential_9/reshape_4/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/Shape?
4sequential_9/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_16/strided_slice/stack?
6sequential_9/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_1?
6sequential_9/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_2?
.sequential_9/conv2d_transpose_16/strided_sliceStridedSlice/sequential_9/conv2d_transpose_16/Shape:output:0=sequential_9/conv2d_transpose_16/strided_slice/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_16/strided_slice?
(sequential_9/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/1?
(sequential_9/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/2?
(sequential_9/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_9/conv2d_transpose_16/stack/3?
&sequential_9/conv2d_transpose_16/stackPack7sequential_9/conv2d_transpose_16/strided_slice:output:01sequential_9/conv2d_transpose_16/stack/1:output:01sequential_9/conv2d_transpose_16/stack/2:output:01sequential_9/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/stack?
6sequential_9/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_16/strided_slice_1/stack?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_16/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_16/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_16/strided_slice_1?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_16/stack:output:0Hsequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0'sequential_9/reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_16/conv2d_transpose?
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_16/BiasAddBiasAdd:sequential_9/conv2d_transpose_16/conv2d_transpose:output:0?sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2*
(sequential_9/conv2d_transpose_16/BiasAdd?
2sequential_9/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_9/batch_normalization_26/ReadVariableOp?
4sequential_9/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_9/batch_normalization_26/ReadVariableOp_1?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_16/BiasAdd:output:0:sequential_9/batch_normalization_26/ReadVariableOp:value:0<sequential_9/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_26/FusedBatchNormV3?
2sequential_9/batch_normalization_26/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_26/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_26/AssignNewValue?
4sequential_9/batch_normalization_26/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_26/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_26/AssignNewValue_1?
sequential_9/activation_40/ReluRelu8sequential_9/batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_9/activation_40/Relu?
&sequential_9/conv2d_transpose_17/ShapeShape-sequential_9/activation_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/Shape?
4sequential_9/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_17/strided_slice/stack?
6sequential_9/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_1?
6sequential_9/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_2?
.sequential_9/conv2d_transpose_17/strided_sliceStridedSlice/sequential_9/conv2d_transpose_17/Shape:output:0=sequential_9/conv2d_transpose_17/strided_slice/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_17/strided_slice?
(sequential_9/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/1?
(sequential_9/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/2?
(sequential_9/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_17/stack/3?
&sequential_9/conv2d_transpose_17/stackPack7sequential_9/conv2d_transpose_17/strided_slice:output:01sequential_9/conv2d_transpose_17/stack/1:output:01sequential_9/conv2d_transpose_17/stack/2:output:01sequential_9/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/stack?
6sequential_9/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_17/strided_slice_1/stack?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_17/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_17/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_17/strided_slice_1?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_17/stack:output:0Hsequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_17/conv2d_transpose?
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_17/BiasAddBiasAdd:sequential_9/conv2d_transpose_17/conv2d_transpose:output:0?sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_17/BiasAdd?
2sequential_9/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_9/batch_normalization_27/ReadVariableOp?
4sequential_9/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_9/batch_normalization_27/ReadVariableOp_1?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_27/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_17/BiasAdd:output:0:sequential_9/batch_normalization_27/ReadVariableOp:value:0<sequential_9/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_27/FusedBatchNormV3?
2sequential_9/batch_normalization_27/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_27/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_27/AssignNewValue?
4sequential_9/batch_normalization_27/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_27/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_27/AssignNewValue_1?
sequential_9/activation_41/ReluRelu8sequential_9/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_9/activation_41/Relu?
&sequential_9/conv2d_transpose_18/ShapeShape-sequential_9/activation_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/Shape?
4sequential_9/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_18/strided_slice/stack?
6sequential_9/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_1?
6sequential_9/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_2?
.sequential_9/conv2d_transpose_18/strided_sliceStridedSlice/sequential_9/conv2d_transpose_18/Shape:output:0=sequential_9/conv2d_transpose_18/strided_slice/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_18/strided_slice?
(sequential_9/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/1?
(sequential_9/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/2?
(sequential_9/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/3?
&sequential_9/conv2d_transpose_18/stackPack7sequential_9/conv2d_transpose_18/strided_slice:output:01sequential_9/conv2d_transpose_18/stack/1:output:01sequential_9/conv2d_transpose_18/stack/2:output:01sequential_9/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/stack?
6sequential_9/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_18/strided_slice_1/stack?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_18/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_18/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_18/strided_slice_1?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_18/stack:output:0Hsequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_18/conv2d_transpose?
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_18/BiasAddBiasAdd:sequential_9/conv2d_transpose_18/conv2d_transpose:output:0?sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2*
(sequential_9/conv2d_transpose_18/BiasAdd?
2sequential_9/batch_normalization_28/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_9/batch_normalization_28/ReadVariableOp?
4sequential_9/batch_normalization_28/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_9/batch_normalization_28/ReadVariableOp_1?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_28/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_18/BiasAdd:output:0:sequential_9/batch_normalization_28/ReadVariableOp:value:0<sequential_9/batch_normalization_28/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_28/FusedBatchNormV3?
2sequential_9/batch_normalization_28/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_28/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_28/AssignNewValue?
4sequential_9/batch_normalization_28/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_28/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_28/AssignNewValue_1?
sequential_9/activation_42/ReluRelu8sequential_9/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2!
sequential_9/activation_42/Relu?
&sequential_9/conv2d_transpose_19/ShapeShape-sequential_9/activation_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/Shape?
4sequential_9/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_19/strided_slice/stack?
6sequential_9/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_1?
6sequential_9/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_2?
.sequential_9/conv2d_transpose_19/strided_sliceStridedSlice/sequential_9/conv2d_transpose_19/Shape:output:0=sequential_9/conv2d_transpose_19/strided_slice/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_19/strided_slice?
(sequential_9/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/1?
(sequential_9/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/2?
(sequential_9/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_19/stack/3?
&sequential_9/conv2d_transpose_19/stackPack7sequential_9/conv2d_transpose_19/strided_slice:output:01sequential_9/conv2d_transpose_19/stack/1:output:01sequential_9/conv2d_transpose_19/stack/2:output:01sequential_9/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/stack?
6sequential_9/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_19/strided_slice_1/stack?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_19/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_19/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_19/strided_slice_1?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_19/stack:output:0Hsequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
23
1sequential_9/conv2d_transpose_19/conv2d_transpose?
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_19/BiasAddBiasAdd:sequential_9/conv2d_transpose_19/conv2d_transpose:output:0?sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2*
(sequential_9/conv2d_transpose_19/BiasAdd?
"sequential_9/activation_43/SigmoidSigmoid1sequential_9/conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"sequential_9/activation_43/Sigmoid?
IdentityIdentity&sequential_9/activation_43/Sigmoid:y:03^sequential_8/batch_normalization_23/AssignNewValue5^sequential_8/batch_normalization_23/AssignNewValue_1D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_23/ReadVariableOp5^sequential_8/batch_normalization_23/ReadVariableOp_13^sequential_8/batch_normalization_24/AssignNewValue5^sequential_8/batch_normalization_24/AssignNewValue_1D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_24/ReadVariableOp5^sequential_8/batch_normalization_24/ReadVariableOp_13^sequential_8/batch_normalization_25/AssignNewValue5^sequential_8/batch_normalization_25/AssignNewValue_1D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_25/ReadVariableOp5^sequential_8/batch_normalization_25/ReadVariableOp_1.^sequential_8/conv2d_11/BiasAdd/ReadVariableOp-^sequential_8/conv2d_11/Conv2D/ReadVariableOp.^sequential_8/conv2d_12/BiasAdd/ReadVariableOp-^sequential_8/conv2d_12/Conv2D/ReadVariableOp.^sequential_8/conv2d_13/BiasAdd/ReadVariableOp-^sequential_8/conv2d_13/Conv2D/ReadVariableOp-^sequential_8/dense_12/BiasAdd/ReadVariableOp,^sequential_8/dense_12/MatMul/ReadVariableOp-^sequential_8/dense_13/BiasAdd/ReadVariableOp,^sequential_8/dense_13/MatMul/ReadVariableOp3^sequential_9/batch_normalization_26/AssignNewValue5^sequential_9/batch_normalization_26/AssignNewValue_1D^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_26/ReadVariableOp5^sequential_9/batch_normalization_26/ReadVariableOp_13^sequential_9/batch_normalization_27/AssignNewValue5^sequential_9/batch_normalization_27/AssignNewValue_1D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_27/ReadVariableOp5^sequential_9/batch_normalization_27/ReadVariableOp_13^sequential_9/batch_normalization_28/AssignNewValue5^sequential_9/batch_normalization_28/AssignNewValue_1D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_28/ReadVariableOp5^sequential_9/batch_normalization_28/ReadVariableOp_18^sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp-^sequential_9/dense_14/BiasAdd/ReadVariableOp,^sequential_9/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2h
2sequential_8/batch_normalization_23/AssignNewValue2sequential_8/batch_normalization_23/AssignNewValue2l
4sequential_8/batch_normalization_23/AssignNewValue_14sequential_8/batch_normalization_23/AssignNewValue_12?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_23/ReadVariableOp2sequential_8/batch_normalization_23/ReadVariableOp2l
4sequential_8/batch_normalization_23/ReadVariableOp_14sequential_8/batch_normalization_23/ReadVariableOp_12h
2sequential_8/batch_normalization_24/AssignNewValue2sequential_8/batch_normalization_24/AssignNewValue2l
4sequential_8/batch_normalization_24/AssignNewValue_14sequential_8/batch_normalization_24/AssignNewValue_12?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_24/ReadVariableOp2sequential_8/batch_normalization_24/ReadVariableOp2l
4sequential_8/batch_normalization_24/ReadVariableOp_14sequential_8/batch_normalization_24/ReadVariableOp_12h
2sequential_8/batch_normalization_25/AssignNewValue2sequential_8/batch_normalization_25/AssignNewValue2l
4sequential_8/batch_normalization_25/AssignNewValue_14sequential_8/batch_normalization_25/AssignNewValue_12?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_25/ReadVariableOp2sequential_8/batch_normalization_25/ReadVariableOp2l
4sequential_8/batch_normalization_25/ReadVariableOp_14sequential_8/batch_normalization_25/ReadVariableOp_12^
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp-sequential_8/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_11/Conv2D/ReadVariableOp,sequential_8/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp-sequential_8/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_12/Conv2D/ReadVariableOp,sequential_8/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp-sequential_8/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_13/Conv2D/ReadVariableOp,sequential_8/conv2d_13/Conv2D/ReadVariableOp2\
,sequential_8/dense_12/BiasAdd/ReadVariableOp,sequential_8/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_12/MatMul/ReadVariableOp+sequential_8/dense_12/MatMul/ReadVariableOp2\
,sequential_8/dense_13/BiasAdd/ReadVariableOp,sequential_8/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_13/MatMul/ReadVariableOp+sequential_8/dense_13/MatMul/ReadVariableOp2h
2sequential_9/batch_normalization_26/AssignNewValue2sequential_9/batch_normalization_26/AssignNewValue2l
4sequential_9/batch_normalization_26/AssignNewValue_14sequential_9/batch_normalization_26/AssignNewValue_12?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_26/ReadVariableOp2sequential_9/batch_normalization_26/ReadVariableOp2l
4sequential_9/batch_normalization_26/ReadVariableOp_14sequential_9/batch_normalization_26/ReadVariableOp_12h
2sequential_9/batch_normalization_27/AssignNewValue2sequential_9/batch_normalization_27/AssignNewValue2l
4sequential_9/batch_normalization_27/AssignNewValue_14sequential_9/batch_normalization_27/AssignNewValue_12?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_27/ReadVariableOp2sequential_9/batch_normalization_27/ReadVariableOp2l
4sequential_9/batch_normalization_27/ReadVariableOp_14sequential_9/batch_normalization_27/ReadVariableOp_12h
2sequential_9/batch_normalization_28/AssignNewValue2sequential_9/batch_normalization_28/AssignNewValue2l
4sequential_9/batch_normalization_28/AssignNewValue_14sequential_9/batch_normalization_28/AssignNewValue_12?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_28/ReadVariableOp2sequential_9/batch_normalization_28/ReadVariableOp2l
4sequential_9/batch_normalization_28/ReadVariableOp_14sequential_9/batch_normalization_28/ReadVariableOp_12r
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2\
,sequential_9/dense_14/BiasAdd/ReadVariableOp,sequential_9/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_14/MatMul/ReadVariableOp+sequential_9/dense_14/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313572

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?2
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311374
input_19
5sequential_8_conv2d_11_conv2d_readvariableop_resource:
6sequential_8_conv2d_11_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_23_readvariableop_resourceA
=sequential_8_batch_normalization_23_readvariableop_1_resourceP
Lsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_12_conv2d_readvariableop_resource:
6sequential_8_conv2d_12_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_24_readvariableop_resourceA
=sequential_8_batch_normalization_24_readvariableop_1_resourceP
Lsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_13_conv2d_readvariableop_resource:
6sequential_8_conv2d_13_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_25_readvariableop_resourceA
=sequential_8_batch_normalization_25_readvariableop_1_resourceP
Lsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8
4sequential_8_dense_12_matmul_readvariableop_resource9
5sequential_8_dense_12_biasadd_readvariableop_resource8
4sequential_8_dense_13_matmul_readvariableop_resource9
5sequential_8_dense_13_biasadd_readvariableop_resource8
4sequential_9_dense_14_matmul_readvariableop_resource9
5sequential_9_dense_14_biasadd_readvariableop_resourceM
Isequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_26_readvariableop_resourceA
=sequential_9_batch_normalization_26_readvariableop_1_resourceP
Lsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_27_readvariableop_resourceA
=sequential_9_batch_normalization_27_readvariableop_1_resourceP
Lsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_28_readvariableop_resourceA
=sequential_9_batch_normalization_28_readvariableop_1_resourceP
Lsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource
identity??2sequential_8/batch_normalization_23/AssignNewValue?4sequential_8/batch_normalization_23/AssignNewValue_1?Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_23/ReadVariableOp?4sequential_8/batch_normalization_23/ReadVariableOp_1?2sequential_8/batch_normalization_24/AssignNewValue?4sequential_8/batch_normalization_24/AssignNewValue_1?Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_24/ReadVariableOp?4sequential_8/batch_normalization_24/ReadVariableOp_1?2sequential_8/batch_normalization_25/AssignNewValue?4sequential_8/batch_normalization_25/AssignNewValue_1?Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_25/ReadVariableOp?4sequential_8/batch_normalization_25/ReadVariableOp_1?-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?,sequential_8/conv2d_11/Conv2D/ReadVariableOp?-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?,sequential_8/conv2d_12/Conv2D/ReadVariableOp?-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?,sequential_8/conv2d_13/Conv2D/ReadVariableOp?,sequential_8/dense_12/BiasAdd/ReadVariableOp?+sequential_8/dense_12/MatMul/ReadVariableOp?,sequential_8/dense_13/BiasAdd/ReadVariableOp?+sequential_8/dense_13/MatMul/ReadVariableOp?2sequential_9/batch_normalization_26/AssignNewValue?4sequential_9/batch_normalization_26/AssignNewValue_1?Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_26/ReadVariableOp?4sequential_9/batch_normalization_26/ReadVariableOp_1?2sequential_9/batch_normalization_27/AssignNewValue?4sequential_9/batch_normalization_27/AssignNewValue_1?Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_27/ReadVariableOp?4sequential_9/batch_normalization_27/ReadVariableOp_1?2sequential_9/batch_normalization_28/AssignNewValue?4sequential_9/batch_normalization_28/AssignNewValue_1?Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_28/ReadVariableOp?4sequential_9/batch_normalization_28/ReadVariableOp_1?7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?,sequential_9/dense_14/BiasAdd/ReadVariableOp?+sequential_9/dense_14/MatMul/ReadVariableOp?
,sequential_8/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_11/Conv2D/ReadVariableOp?
sequential_8/conv2d_11/Conv2DConv2Dinput_14sequential_8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_8/conv2d_11/Conv2D?
-sequential_8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?
sequential_8/conv2d_11/BiasAddBiasAdd&sequential_8/conv2d_11/Conv2D:output:05sequential_8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2 
sequential_8/conv2d_11/BiasAdd?
2sequential_8/batch_normalization_23/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_8/batch_normalization_23/ReadVariableOp?
4sequential_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_8/batch_normalization_23/ReadVariableOp_1?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_11/BiasAdd:output:0:sequential_8/batch_normalization_23/ReadVariableOp:value:0<sequential_8/batch_normalization_23/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_23/FusedBatchNormV3?
2sequential_8/batch_normalization_23/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_23/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_23/AssignNewValue?
4sequential_8/batch_normalization_23/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_23/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_23/AssignNewValue_1?
sequential_8/activation_35/ReluRelu8sequential_8/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2!
sequential_8/activation_35/Relu?
,sequential_8/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_8/conv2d_12/Conv2D/ReadVariableOp?
sequential_8/conv2d_12/Conv2DConv2D-sequential_8/activation_35/Relu:activations:04sequential_8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_8/conv2d_12/Conv2D?
-sequential_8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?
sequential_8/conv2d_12/BiasAddBiasAdd&sequential_8/conv2d_12/Conv2D:output:05sequential_8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_8/conv2d_12/BiasAdd?
2sequential_8/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_8/batch_normalization_24/ReadVariableOp?
4sequential_8/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_8/batch_normalization_24/ReadVariableOp_1?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_12/BiasAdd:output:0:sequential_8/batch_normalization_24/ReadVariableOp:value:0<sequential_8/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_24/FusedBatchNormV3?
2sequential_8/batch_normalization_24/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_24/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_24/AssignNewValue?
4sequential_8/batch_normalization_24/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_24/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_24/AssignNewValue_1?
sequential_8/activation_36/ReluRelu8sequential_8/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_8/activation_36/Relu?
,sequential_8/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02.
,sequential_8/conv2d_13/Conv2D/ReadVariableOp?
sequential_8/conv2d_13/Conv2DConv2D-sequential_8/activation_36/Relu:activations:04sequential_8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
sequential_8/conv2d_13/Conv2D?
-sequential_8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?
sequential_8/conv2d_13/BiasAddBiasAdd&sequential_8/conv2d_13/Conv2D:output:05sequential_8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_13/BiasAdd?
2sequential_8/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_25/ReadVariableOp?
4sequential_8/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_25/ReadVariableOp_1?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_13/BiasAdd:output:0:sequential_8/batch_normalization_25/ReadVariableOp:value:0<sequential_8/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_8/batch_normalization_25/FusedBatchNormV3?
2sequential_8/batch_normalization_25/AssignNewValueAssignVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceAsequential_8/batch_normalization_25/FusedBatchNormV3:batch_mean:0D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_8/batch_normalization_25/AssignNewValue?
4sequential_8/batch_normalization_25/AssignNewValue_1AssignVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceEsequential_8/batch_normalization_25/FusedBatchNormV3:batch_variance:0F^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_8/batch_normalization_25/AssignNewValue_1?
sequential_8/activation_37/ReluRelu8sequential_8/batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_37/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape-sequential_8/activation_37/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
+sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_8/dense_12/MatMul/ReadVariableOp?
sequential_8/dense_12/MatMulMatMul'sequential_8/flatten_4/Reshape:output:03sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/MatMul?
,sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_8/dense_12/BiasAdd/ReadVariableOp?
sequential_8/dense_12/BiasAddBiasAdd&sequential_8/dense_12/MatMul:product:04sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/BiasAdd?
sequential_8/activation_38/ReluRelu&sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_8/activation_38/Relu?
+sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_8/dense_13/MatMul/ReadVariableOp?
sequential_8/dense_13/MatMulMatMul-sequential_8/activation_38/Relu:activations:03sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/MatMul?
,sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_8/dense_13/BiasAdd/ReadVariableOp?
sequential_8/dense_13/BiasAddBiasAdd&sequential_8/dense_13/MatMul:product:04sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/BiasAdd?
+sequential_9/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02-
+sequential_9/dense_14/MatMul/ReadVariableOp?
sequential_9/dense_14/MatMulMatMul&sequential_8/dense_13/BiasAdd:output:03sequential_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/MatMul?
,sequential_9/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02.
,sequential_9/dense_14/BiasAdd/ReadVariableOp?
sequential_9/dense_14/BiasAddBiasAdd&sequential_9/dense_14/MatMul:product:04sequential_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/BiasAdd?
sequential_9/activation_39/ReluRelu&sequential_9/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2!
sequential_9/activation_39/Relu?
sequential_9/reshape_4/ShapeShape-sequential_9/activation_39/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_4/Shape?
*sequential_9/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/reshape_4/strided_slice/stack?
,sequential_9/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_1?
,sequential_9/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_2?
$sequential_9/reshape_4/strided_sliceStridedSlice%sequential_9/reshape_4/Shape:output:03sequential_9/reshape_4/strided_slice/stack:output:05sequential_9/reshape_4/strided_slice/stack_1:output:05sequential_9/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/reshape_4/strided_slice?
&sequential_9/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/1?
&sequential_9/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/2?
&sequential_9/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_9/reshape_4/Reshape/shape/3?
$sequential_9/reshape_4/Reshape/shapePack-sequential_9/reshape_4/strided_slice:output:0/sequential_9/reshape_4/Reshape/shape/1:output:0/sequential_9/reshape_4/Reshape/shape/2:output:0/sequential_9/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/reshape_4/Reshape/shape?
sequential_9/reshape_4/ReshapeReshape-sequential_9/activation_39/Relu:activations:0-sequential_9/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2 
sequential_9/reshape_4/Reshape?
&sequential_9/conv2d_transpose_16/ShapeShape'sequential_9/reshape_4/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/Shape?
4sequential_9/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_16/strided_slice/stack?
6sequential_9/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_1?
6sequential_9/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_2?
.sequential_9/conv2d_transpose_16/strided_sliceStridedSlice/sequential_9/conv2d_transpose_16/Shape:output:0=sequential_9/conv2d_transpose_16/strided_slice/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_16/strided_slice?
(sequential_9/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/1?
(sequential_9/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/2?
(sequential_9/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_9/conv2d_transpose_16/stack/3?
&sequential_9/conv2d_transpose_16/stackPack7sequential_9/conv2d_transpose_16/strided_slice:output:01sequential_9/conv2d_transpose_16/stack/1:output:01sequential_9/conv2d_transpose_16/stack/2:output:01sequential_9/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/stack?
6sequential_9/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_16/strided_slice_1/stack?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_16/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_16/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_16/strided_slice_1?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_16/stack:output:0Hsequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0'sequential_9/reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_16/conv2d_transpose?
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_16/BiasAddBiasAdd:sequential_9/conv2d_transpose_16/conv2d_transpose:output:0?sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2*
(sequential_9/conv2d_transpose_16/BiasAdd?
2sequential_9/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_9/batch_normalization_26/ReadVariableOp?
4sequential_9/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_9/batch_normalization_26/ReadVariableOp_1?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_16/BiasAdd:output:0:sequential_9/batch_normalization_26/ReadVariableOp:value:0<sequential_9/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_26/FusedBatchNormV3?
2sequential_9/batch_normalization_26/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_26/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_26/AssignNewValue?
4sequential_9/batch_normalization_26/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_26/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_26/AssignNewValue_1?
sequential_9/activation_40/ReluRelu8sequential_9/batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_9/activation_40/Relu?
&sequential_9/conv2d_transpose_17/ShapeShape-sequential_9/activation_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/Shape?
4sequential_9/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_17/strided_slice/stack?
6sequential_9/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_1?
6sequential_9/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_2?
.sequential_9/conv2d_transpose_17/strided_sliceStridedSlice/sequential_9/conv2d_transpose_17/Shape:output:0=sequential_9/conv2d_transpose_17/strided_slice/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_17/strided_slice?
(sequential_9/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/1?
(sequential_9/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/2?
(sequential_9/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_17/stack/3?
&sequential_9/conv2d_transpose_17/stackPack7sequential_9/conv2d_transpose_17/strided_slice:output:01sequential_9/conv2d_transpose_17/stack/1:output:01sequential_9/conv2d_transpose_17/stack/2:output:01sequential_9/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/stack?
6sequential_9/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_17/strided_slice_1/stack?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_17/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_17/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_17/strided_slice_1?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_17/stack:output:0Hsequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_17/conv2d_transpose?
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_17/BiasAddBiasAdd:sequential_9/conv2d_transpose_17/conv2d_transpose:output:0?sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_17/BiasAdd?
2sequential_9/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_9/batch_normalization_27/ReadVariableOp?
4sequential_9/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_9/batch_normalization_27/ReadVariableOp_1?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_27/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_17/BiasAdd:output:0:sequential_9/batch_normalization_27/ReadVariableOp:value:0<sequential_9/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_27/FusedBatchNormV3?
2sequential_9/batch_normalization_27/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_27/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_27/AssignNewValue?
4sequential_9/batch_normalization_27/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_27/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_27/AssignNewValue_1?
sequential_9/activation_41/ReluRelu8sequential_9/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_9/activation_41/Relu?
&sequential_9/conv2d_transpose_18/ShapeShape-sequential_9/activation_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/Shape?
4sequential_9/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_18/strided_slice/stack?
6sequential_9/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_1?
6sequential_9/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_2?
.sequential_9/conv2d_transpose_18/strided_sliceStridedSlice/sequential_9/conv2d_transpose_18/Shape:output:0=sequential_9/conv2d_transpose_18/strided_slice/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_18/strided_slice?
(sequential_9/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/1?
(sequential_9/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/2?
(sequential_9/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/3?
&sequential_9/conv2d_transpose_18/stackPack7sequential_9/conv2d_transpose_18/strided_slice:output:01sequential_9/conv2d_transpose_18/stack/1:output:01sequential_9/conv2d_transpose_18/stack/2:output:01sequential_9/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/stack?
6sequential_9/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_18/strided_slice_1/stack?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_18/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_18/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_18/strided_slice_1?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_18/stack:output:0Hsequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_18/conv2d_transpose?
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_18/BiasAddBiasAdd:sequential_9/conv2d_transpose_18/conv2d_transpose:output:0?sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2*
(sequential_9/conv2d_transpose_18/BiasAdd?
2sequential_9/batch_normalization_28/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_9/batch_normalization_28/ReadVariableOp?
4sequential_9/batch_normalization_28/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_9/batch_normalization_28/ReadVariableOp_1?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_28/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_18/BiasAdd:output:0:sequential_9/batch_normalization_28/ReadVariableOp:value:0<sequential_9/batch_normalization_28/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<26
4sequential_9/batch_normalization_28/FusedBatchNormV3?
2sequential_9/batch_normalization_28/AssignNewValueAssignVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceAsequential_9/batch_normalization_28/FusedBatchNormV3:batch_mean:0D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*_
_classU
SQloc:@sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype024
2sequential_9/batch_normalization_28/AssignNewValue?
4sequential_9/batch_normalization_28/AssignNewValue_1AssignVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceEsequential_9/batch_normalization_28/FusedBatchNormV3:batch_variance:0F^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*a
_classW
USloc:@sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype026
4sequential_9/batch_normalization_28/AssignNewValue_1?
sequential_9/activation_42/ReluRelu8sequential_9/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2!
sequential_9/activation_42/Relu?
&sequential_9/conv2d_transpose_19/ShapeShape-sequential_9/activation_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/Shape?
4sequential_9/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_19/strided_slice/stack?
6sequential_9/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_1?
6sequential_9/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_2?
.sequential_9/conv2d_transpose_19/strided_sliceStridedSlice/sequential_9/conv2d_transpose_19/Shape:output:0=sequential_9/conv2d_transpose_19/strided_slice/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_19/strided_slice?
(sequential_9/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/1?
(sequential_9/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/2?
(sequential_9/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_19/stack/3?
&sequential_9/conv2d_transpose_19/stackPack7sequential_9/conv2d_transpose_19/strided_slice:output:01sequential_9/conv2d_transpose_19/stack/1:output:01sequential_9/conv2d_transpose_19/stack/2:output:01sequential_9/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/stack?
6sequential_9/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_19/strided_slice_1/stack?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_19/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_19/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_19/strided_slice_1?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_19/stack:output:0Hsequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
23
1sequential_9/conv2d_transpose_19/conv2d_transpose?
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_19/BiasAddBiasAdd:sequential_9/conv2d_transpose_19/conv2d_transpose:output:0?sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2*
(sequential_9/conv2d_transpose_19/BiasAdd?
"sequential_9/activation_43/SigmoidSigmoid1sequential_9/conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"sequential_9/activation_43/Sigmoid?
IdentityIdentity&sequential_9/activation_43/Sigmoid:y:03^sequential_8/batch_normalization_23/AssignNewValue5^sequential_8/batch_normalization_23/AssignNewValue_1D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_23/ReadVariableOp5^sequential_8/batch_normalization_23/ReadVariableOp_13^sequential_8/batch_normalization_24/AssignNewValue5^sequential_8/batch_normalization_24/AssignNewValue_1D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_24/ReadVariableOp5^sequential_8/batch_normalization_24/ReadVariableOp_13^sequential_8/batch_normalization_25/AssignNewValue5^sequential_8/batch_normalization_25/AssignNewValue_1D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_25/ReadVariableOp5^sequential_8/batch_normalization_25/ReadVariableOp_1.^sequential_8/conv2d_11/BiasAdd/ReadVariableOp-^sequential_8/conv2d_11/Conv2D/ReadVariableOp.^sequential_8/conv2d_12/BiasAdd/ReadVariableOp-^sequential_8/conv2d_12/Conv2D/ReadVariableOp.^sequential_8/conv2d_13/BiasAdd/ReadVariableOp-^sequential_8/conv2d_13/Conv2D/ReadVariableOp-^sequential_8/dense_12/BiasAdd/ReadVariableOp,^sequential_8/dense_12/MatMul/ReadVariableOp-^sequential_8/dense_13/BiasAdd/ReadVariableOp,^sequential_8/dense_13/MatMul/ReadVariableOp3^sequential_9/batch_normalization_26/AssignNewValue5^sequential_9/batch_normalization_26/AssignNewValue_1D^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_26/ReadVariableOp5^sequential_9/batch_normalization_26/ReadVariableOp_13^sequential_9/batch_normalization_27/AssignNewValue5^sequential_9/batch_normalization_27/AssignNewValue_1D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_27/ReadVariableOp5^sequential_9/batch_normalization_27/ReadVariableOp_13^sequential_9/batch_normalization_28/AssignNewValue5^sequential_9/batch_normalization_28/AssignNewValue_1D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_28/ReadVariableOp5^sequential_9/batch_normalization_28/ReadVariableOp_18^sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp-^sequential_9/dense_14/BiasAdd/ReadVariableOp,^sequential_9/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2h
2sequential_8/batch_normalization_23/AssignNewValue2sequential_8/batch_normalization_23/AssignNewValue2l
4sequential_8/batch_normalization_23/AssignNewValue_14sequential_8/batch_normalization_23/AssignNewValue_12?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_23/ReadVariableOp2sequential_8/batch_normalization_23/ReadVariableOp2l
4sequential_8/batch_normalization_23/ReadVariableOp_14sequential_8/batch_normalization_23/ReadVariableOp_12h
2sequential_8/batch_normalization_24/AssignNewValue2sequential_8/batch_normalization_24/AssignNewValue2l
4sequential_8/batch_normalization_24/AssignNewValue_14sequential_8/batch_normalization_24/AssignNewValue_12?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_24/ReadVariableOp2sequential_8/batch_normalization_24/ReadVariableOp2l
4sequential_8/batch_normalization_24/ReadVariableOp_14sequential_8/batch_normalization_24/ReadVariableOp_12h
2sequential_8/batch_normalization_25/AssignNewValue2sequential_8/batch_normalization_25/AssignNewValue2l
4sequential_8/batch_normalization_25/AssignNewValue_14sequential_8/batch_normalization_25/AssignNewValue_12?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_25/ReadVariableOp2sequential_8/batch_normalization_25/ReadVariableOp2l
4sequential_8/batch_normalization_25/ReadVariableOp_14sequential_8/batch_normalization_25/ReadVariableOp_12^
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp-sequential_8/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_11/Conv2D/ReadVariableOp,sequential_8/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp-sequential_8/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_12/Conv2D/ReadVariableOp,sequential_8/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp-sequential_8/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_13/Conv2D/ReadVariableOp,sequential_8/conv2d_13/Conv2D/ReadVariableOp2\
,sequential_8/dense_12/BiasAdd/ReadVariableOp,sequential_8/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_12/MatMul/ReadVariableOp+sequential_8/dense_12/MatMul/ReadVariableOp2\
,sequential_8/dense_13/BiasAdd/ReadVariableOp,sequential_8/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_13/MatMul/ReadVariableOp+sequential_8/dense_13/MatMul/ReadVariableOp2h
2sequential_9/batch_normalization_26/AssignNewValue2sequential_9/batch_normalization_26/AssignNewValue2l
4sequential_9/batch_normalization_26/AssignNewValue_14sequential_9/batch_normalization_26/AssignNewValue_12?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_26/ReadVariableOp2sequential_9/batch_normalization_26/ReadVariableOp2l
4sequential_9/batch_normalization_26/ReadVariableOp_14sequential_9/batch_normalization_26/ReadVariableOp_12h
2sequential_9/batch_normalization_27/AssignNewValue2sequential_9/batch_normalization_27/AssignNewValue2l
4sequential_9/batch_normalization_27/AssignNewValue_14sequential_9/batch_normalization_27/AssignNewValue_12?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_27/ReadVariableOp2sequential_9/batch_normalization_27/ReadVariableOp2l
4sequential_9/batch_normalization_27/ReadVariableOp_14sequential_9/batch_normalization_27/ReadVariableOp_12h
2sequential_9/batch_normalization_28/AssignNewValue2sequential_9/batch_normalization_28/AssignNewValue2l
4sequential_9/batch_normalization_28/AssignNewValue_14sequential_9/batch_normalization_28/AssignNewValue_12?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_28/ReadVariableOp2sequential_9/batch_normalization_28/ReadVariableOp2l
4sequential_9/batch_normalization_28/ReadVariableOp_14sequential_9/batch_normalization_28/ReadVariableOp_12r
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2\
,sequential_9/dense_14/BiasAdd/ReadVariableOp,sequential_9/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_14/MatMul/ReadVariableOp+sequential_9/dense_14/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
7__inference_batch_normalization_24_layer_call_fn_313709

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3088292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_314042

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_16_layer_call_fn_309396

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_3093862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314085

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_38_layer_call_and_return_conditional_losses_309035

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_19_layer_call_fn_309844

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_3098342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_309785

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

*__inference_conv2d_12_layer_call_fn_313645

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_3087762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_37_layer_call_and_return_conditional_losses_313935

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_4_layer_call_fn_313951

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3089962
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_41_layer_call_and_return_conditional_losses_314190

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_18_layer_call_fn_309692

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_3096822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
/__inference_auto_encoder_4_layer_call_fn_311781
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_3108542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_308535

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_23_layer_call_fn_313552

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3087172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_313371

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_18_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_19_biasadd_readvariableop_resource
identity??6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/BiasAdd~
activation_39/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
activation_39/Relur
reshape_4/ShapeShape activation_39/Relu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape activation_39/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape?
conv2d_transpose_16/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_16/BiasAdd?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_16/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3?
activation_40/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_40/Relu?
conv2d_transpose_17/ShapeShape activation_40/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0 activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_17/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_17/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
activation_41/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_41/Relu?
conv2d_transpose_18/ShapeShape activation_41/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice|
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/1|
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0 activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_transpose_18/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_18/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3?
activation_42/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
activation_42/Relu?
conv2d_transpose_19/ShapeShape activation_42/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice|
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/1|
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0 activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_19/BiasAdd?
activation_43/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
activation_43/Sigmoid?	
IdentityIdentityactivation_43/Sigmoid:y:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_activation_41_layer_call_and_return_conditional_losses_310007

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_309386

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_312509

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??%batch_normalization_23/AssignNewValue?'batch_normalization_23/AssignNewValue_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?%batch_normalization_24/AssignNewValue?'batch_normalization_24/AssignNewValue_1?6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?%batch_normalization_25/AssignNewValue?'batch_normalization_25/AssignNewValue_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/BiasAdd?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_23/ReadVariableOp?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_23/ReadVariableOp_1?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_23/FusedBatchNormV3?
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_23/AssignNewValue?
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_23/AssignNewValue_1?
activation_35/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation_35/Relu?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D activation_35/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/BiasAdd?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_24/ReadVariableOp?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_24/ReadVariableOp_1?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_24/FusedBatchNormV3?
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_24/AssignNewValue?
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_24/AssignNewValue_1?
activation_36/ReluRelu+batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_36/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D activation_36/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_13/BiasAdd?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_25/FusedBatchNormV3?
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue?
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1?
activation_37/ReluRelu+batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_37/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshape activation_37/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd~
activation_38/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_38/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul activation_38/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?

IdentityIdentitydense_13/BiasAdd:output:0&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_312689

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3093052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_313636

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_308431

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_36_layer_call_and_return_conditional_losses_308870

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_39_layer_call_and_return_conditional_losses_314023

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:?????????? 2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_309014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313840

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_40_layer_call_and_return_conditional_losses_314116

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?,
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312235
x9
5sequential_8_conv2d_11_conv2d_readvariableop_resource:
6sequential_8_conv2d_11_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_23_readvariableop_resourceA
=sequential_8_batch_normalization_23_readvariableop_1_resourceP
Lsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_12_conv2d_readvariableop_resource:
6sequential_8_conv2d_12_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_24_readvariableop_resourceA
=sequential_8_batch_normalization_24_readvariableop_1_resourceP
Lsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_13_conv2d_readvariableop_resource:
6sequential_8_conv2d_13_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_25_readvariableop_resourceA
=sequential_8_batch_normalization_25_readvariableop_1_resourceP
Lsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8
4sequential_8_dense_12_matmul_readvariableop_resource9
5sequential_8_dense_12_biasadd_readvariableop_resource8
4sequential_8_dense_13_matmul_readvariableop_resource9
5sequential_8_dense_13_biasadd_readvariableop_resource8
4sequential_9_dense_14_matmul_readvariableop_resource9
5sequential_9_dense_14_biasadd_readvariableop_resourceM
Isequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_26_readvariableop_resourceA
=sequential_9_batch_normalization_26_readvariableop_1_resourceP
Lsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_27_readvariableop_resourceA
=sequential_9_batch_normalization_27_readvariableop_1_resourceP
Lsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_28_readvariableop_resourceA
=sequential_9_batch_normalization_28_readvariableop_1_resourceP
Lsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource
identity??Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_23/ReadVariableOp?4sequential_8/batch_normalization_23/ReadVariableOp_1?Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_24/ReadVariableOp?4sequential_8/batch_normalization_24/ReadVariableOp_1?Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_25/ReadVariableOp?4sequential_8/batch_normalization_25/ReadVariableOp_1?-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?,sequential_8/conv2d_11/Conv2D/ReadVariableOp?-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?,sequential_8/conv2d_12/Conv2D/ReadVariableOp?-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?,sequential_8/conv2d_13/Conv2D/ReadVariableOp?,sequential_8/dense_12/BiasAdd/ReadVariableOp?+sequential_8/dense_12/MatMul/ReadVariableOp?,sequential_8/dense_13/BiasAdd/ReadVariableOp?+sequential_8/dense_13/MatMul/ReadVariableOp?Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_26/ReadVariableOp?4sequential_9/batch_normalization_26/ReadVariableOp_1?Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_27/ReadVariableOp?4sequential_9/batch_normalization_27/ReadVariableOp_1?Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_28/ReadVariableOp?4sequential_9/batch_normalization_28/ReadVariableOp_1?7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?,sequential_9/dense_14/BiasAdd/ReadVariableOp?+sequential_9/dense_14/MatMul/ReadVariableOp?
,sequential_8/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_11/Conv2D/ReadVariableOp?
sequential_8/conv2d_11/Conv2DConv2Dx4sequential_8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_8/conv2d_11/Conv2D?
-sequential_8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?
sequential_8/conv2d_11/BiasAddBiasAdd&sequential_8/conv2d_11/Conv2D:output:05sequential_8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2 
sequential_8/conv2d_11/BiasAdd?
2sequential_8/batch_normalization_23/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_8/batch_normalization_23/ReadVariableOp?
4sequential_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_8/batch_normalization_23/ReadVariableOp_1?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_11/BiasAdd:output:0:sequential_8/batch_normalization_23/ReadVariableOp:value:0<sequential_8/batch_normalization_23/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_23/FusedBatchNormV3?
sequential_8/activation_35/ReluRelu8sequential_8/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2!
sequential_8/activation_35/Relu?
,sequential_8/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_8/conv2d_12/Conv2D/ReadVariableOp?
sequential_8/conv2d_12/Conv2DConv2D-sequential_8/activation_35/Relu:activations:04sequential_8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_8/conv2d_12/Conv2D?
-sequential_8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?
sequential_8/conv2d_12/BiasAddBiasAdd&sequential_8/conv2d_12/Conv2D:output:05sequential_8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_8/conv2d_12/BiasAdd?
2sequential_8/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_8/batch_normalization_24/ReadVariableOp?
4sequential_8/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_8/batch_normalization_24/ReadVariableOp_1?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_12/BiasAdd:output:0:sequential_8/batch_normalization_24/ReadVariableOp:value:0<sequential_8/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_24/FusedBatchNormV3?
sequential_8/activation_36/ReluRelu8sequential_8/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_8/activation_36/Relu?
,sequential_8/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02.
,sequential_8/conv2d_13/Conv2D/ReadVariableOp?
sequential_8/conv2d_13/Conv2DConv2D-sequential_8/activation_36/Relu:activations:04sequential_8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
sequential_8/conv2d_13/Conv2D?
-sequential_8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?
sequential_8/conv2d_13/BiasAddBiasAdd&sequential_8/conv2d_13/Conv2D:output:05sequential_8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_13/BiasAdd?
2sequential_8/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_25/ReadVariableOp?
4sequential_8/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_25/ReadVariableOp_1?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_13/BiasAdd:output:0:sequential_8/batch_normalization_25/ReadVariableOp:value:0<sequential_8/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_25/FusedBatchNormV3?
sequential_8/activation_37/ReluRelu8sequential_8/batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_37/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape-sequential_8/activation_37/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
+sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_8/dense_12/MatMul/ReadVariableOp?
sequential_8/dense_12/MatMulMatMul'sequential_8/flatten_4/Reshape:output:03sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/MatMul?
,sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_8/dense_12/BiasAdd/ReadVariableOp?
sequential_8/dense_12/BiasAddBiasAdd&sequential_8/dense_12/MatMul:product:04sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/BiasAdd?
sequential_8/activation_38/ReluRelu&sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_8/activation_38/Relu?
+sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_8/dense_13/MatMul/ReadVariableOp?
sequential_8/dense_13/MatMulMatMul-sequential_8/activation_38/Relu:activations:03sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/MatMul?
,sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_8/dense_13/BiasAdd/ReadVariableOp?
sequential_8/dense_13/BiasAddBiasAdd&sequential_8/dense_13/MatMul:product:04sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/BiasAdd?
+sequential_9/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02-
+sequential_9/dense_14/MatMul/ReadVariableOp?
sequential_9/dense_14/MatMulMatMul&sequential_8/dense_13/BiasAdd:output:03sequential_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/MatMul?
,sequential_9/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02.
,sequential_9/dense_14/BiasAdd/ReadVariableOp?
sequential_9/dense_14/BiasAddBiasAdd&sequential_9/dense_14/MatMul:product:04sequential_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/BiasAdd?
sequential_9/activation_39/ReluRelu&sequential_9/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2!
sequential_9/activation_39/Relu?
sequential_9/reshape_4/ShapeShape-sequential_9/activation_39/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_4/Shape?
*sequential_9/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/reshape_4/strided_slice/stack?
,sequential_9/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_1?
,sequential_9/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_2?
$sequential_9/reshape_4/strided_sliceStridedSlice%sequential_9/reshape_4/Shape:output:03sequential_9/reshape_4/strided_slice/stack:output:05sequential_9/reshape_4/strided_slice/stack_1:output:05sequential_9/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/reshape_4/strided_slice?
&sequential_9/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/1?
&sequential_9/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/2?
&sequential_9/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_9/reshape_4/Reshape/shape/3?
$sequential_9/reshape_4/Reshape/shapePack-sequential_9/reshape_4/strided_slice:output:0/sequential_9/reshape_4/Reshape/shape/1:output:0/sequential_9/reshape_4/Reshape/shape/2:output:0/sequential_9/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/reshape_4/Reshape/shape?
sequential_9/reshape_4/ReshapeReshape-sequential_9/activation_39/Relu:activations:0-sequential_9/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2 
sequential_9/reshape_4/Reshape?
&sequential_9/conv2d_transpose_16/ShapeShape'sequential_9/reshape_4/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/Shape?
4sequential_9/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_16/strided_slice/stack?
6sequential_9/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_1?
6sequential_9/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_2?
.sequential_9/conv2d_transpose_16/strided_sliceStridedSlice/sequential_9/conv2d_transpose_16/Shape:output:0=sequential_9/conv2d_transpose_16/strided_slice/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_16/strided_slice?
(sequential_9/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/1?
(sequential_9/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/2?
(sequential_9/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_9/conv2d_transpose_16/stack/3?
&sequential_9/conv2d_transpose_16/stackPack7sequential_9/conv2d_transpose_16/strided_slice:output:01sequential_9/conv2d_transpose_16/stack/1:output:01sequential_9/conv2d_transpose_16/stack/2:output:01sequential_9/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/stack?
6sequential_9/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_16/strided_slice_1/stack?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_16/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_16/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_16/strided_slice_1?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_16/stack:output:0Hsequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0'sequential_9/reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_16/conv2d_transpose?
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_16/BiasAddBiasAdd:sequential_9/conv2d_transpose_16/conv2d_transpose:output:0?sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2*
(sequential_9/conv2d_transpose_16/BiasAdd?
2sequential_9/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_9/batch_normalization_26/ReadVariableOp?
4sequential_9/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_9/batch_normalization_26/ReadVariableOp_1?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_16/BiasAdd:output:0:sequential_9/batch_normalization_26/ReadVariableOp:value:0<sequential_9/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_26/FusedBatchNormV3?
sequential_9/activation_40/ReluRelu8sequential_9/batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_9/activation_40/Relu?
&sequential_9/conv2d_transpose_17/ShapeShape-sequential_9/activation_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/Shape?
4sequential_9/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_17/strided_slice/stack?
6sequential_9/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_1?
6sequential_9/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_2?
.sequential_9/conv2d_transpose_17/strided_sliceStridedSlice/sequential_9/conv2d_transpose_17/Shape:output:0=sequential_9/conv2d_transpose_17/strided_slice/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_17/strided_slice?
(sequential_9/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/1?
(sequential_9/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/2?
(sequential_9/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_17/stack/3?
&sequential_9/conv2d_transpose_17/stackPack7sequential_9/conv2d_transpose_17/strided_slice:output:01sequential_9/conv2d_transpose_17/stack/1:output:01sequential_9/conv2d_transpose_17/stack/2:output:01sequential_9/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/stack?
6sequential_9/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_17/strided_slice_1/stack?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_17/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_17/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_17/strided_slice_1?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_17/stack:output:0Hsequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_17/conv2d_transpose?
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_17/BiasAddBiasAdd:sequential_9/conv2d_transpose_17/conv2d_transpose:output:0?sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_17/BiasAdd?
2sequential_9/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_9/batch_normalization_27/ReadVariableOp?
4sequential_9/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_9/batch_normalization_27/ReadVariableOp_1?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_27/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_17/BiasAdd:output:0:sequential_9/batch_normalization_27/ReadVariableOp:value:0<sequential_9/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_27/FusedBatchNormV3?
sequential_9/activation_41/ReluRelu8sequential_9/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_9/activation_41/Relu?
&sequential_9/conv2d_transpose_18/ShapeShape-sequential_9/activation_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/Shape?
4sequential_9/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_18/strided_slice/stack?
6sequential_9/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_1?
6sequential_9/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_2?
.sequential_9/conv2d_transpose_18/strided_sliceStridedSlice/sequential_9/conv2d_transpose_18/Shape:output:0=sequential_9/conv2d_transpose_18/strided_slice/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_18/strided_slice?
(sequential_9/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/1?
(sequential_9/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/2?
(sequential_9/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/3?
&sequential_9/conv2d_transpose_18/stackPack7sequential_9/conv2d_transpose_18/strided_slice:output:01sequential_9/conv2d_transpose_18/stack/1:output:01sequential_9/conv2d_transpose_18/stack/2:output:01sequential_9/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/stack?
6sequential_9/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_18/strided_slice_1/stack?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_18/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_18/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_18/strided_slice_1?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_18/stack:output:0Hsequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_18/conv2d_transpose?
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_18/BiasAddBiasAdd:sequential_9/conv2d_transpose_18/conv2d_transpose:output:0?sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2*
(sequential_9/conv2d_transpose_18/BiasAdd?
2sequential_9/batch_normalization_28/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_9/batch_normalization_28/ReadVariableOp?
4sequential_9/batch_normalization_28/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_9/batch_normalization_28/ReadVariableOp_1?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_28/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_18/BiasAdd:output:0:sequential_9/batch_normalization_28/ReadVariableOp:value:0<sequential_9/batch_normalization_28/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_28/FusedBatchNormV3?
sequential_9/activation_42/ReluRelu8sequential_9/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2!
sequential_9/activation_42/Relu?
&sequential_9/conv2d_transpose_19/ShapeShape-sequential_9/activation_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/Shape?
4sequential_9/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_19/strided_slice/stack?
6sequential_9/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_1?
6sequential_9/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_2?
.sequential_9/conv2d_transpose_19/strided_sliceStridedSlice/sequential_9/conv2d_transpose_19/Shape:output:0=sequential_9/conv2d_transpose_19/strided_slice/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_19/strided_slice?
(sequential_9/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/1?
(sequential_9/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/2?
(sequential_9/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_19/stack/3?
&sequential_9/conv2d_transpose_19/stackPack7sequential_9/conv2d_transpose_19/strided_slice:output:01sequential_9/conv2d_transpose_19/stack/1:output:01sequential_9/conv2d_transpose_19/stack/2:output:01sequential_9/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/stack?
6sequential_9/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_19/strided_slice_1/stack?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_19/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_19/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_19/strided_slice_1?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_19/stack:output:0Hsequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
23
1sequential_9/conv2d_transpose_19/conv2d_transpose?
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_19/BiasAddBiasAdd:sequential_9/conv2d_transpose_19/conv2d_transpose:output:0?sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2*
(sequential_9/conv2d_transpose_19/BiasAdd?
"sequential_9/activation_43/SigmoidSigmoid1sequential_9/conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"sequential_9/activation_43/Sigmoid?
IdentityIdentity&sequential_9/activation_43/Sigmoid:y:0D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_23/ReadVariableOp5^sequential_8/batch_normalization_23/ReadVariableOp_1D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_24/ReadVariableOp5^sequential_8/batch_normalization_24/ReadVariableOp_1D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_25/ReadVariableOp5^sequential_8/batch_normalization_25/ReadVariableOp_1.^sequential_8/conv2d_11/BiasAdd/ReadVariableOp-^sequential_8/conv2d_11/Conv2D/ReadVariableOp.^sequential_8/conv2d_12/BiasAdd/ReadVariableOp-^sequential_8/conv2d_12/Conv2D/ReadVariableOp.^sequential_8/conv2d_13/BiasAdd/ReadVariableOp-^sequential_8/conv2d_13/Conv2D/ReadVariableOp-^sequential_8/dense_12/BiasAdd/ReadVariableOp,^sequential_8/dense_12/MatMul/ReadVariableOp-^sequential_8/dense_13/BiasAdd/ReadVariableOp,^sequential_8/dense_13/MatMul/ReadVariableOpD^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_26/ReadVariableOp5^sequential_9/batch_normalization_26/ReadVariableOp_1D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_27/ReadVariableOp5^sequential_9/batch_normalization_27/ReadVariableOp_1D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_28/ReadVariableOp5^sequential_9/batch_normalization_28/ReadVariableOp_18^sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp-^sequential_9/dense_14/BiasAdd/ReadVariableOp,^sequential_9/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_23/ReadVariableOp2sequential_8/batch_normalization_23/ReadVariableOp2l
4sequential_8/batch_normalization_23/ReadVariableOp_14sequential_8/batch_normalization_23/ReadVariableOp_12?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_24/ReadVariableOp2sequential_8/batch_normalization_24/ReadVariableOp2l
4sequential_8/batch_normalization_24/ReadVariableOp_14sequential_8/batch_normalization_24/ReadVariableOp_12?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_25/ReadVariableOp2sequential_8/batch_normalization_25/ReadVariableOp2l
4sequential_8/batch_normalization_25/ReadVariableOp_14sequential_8/batch_normalization_25/ReadVariableOp_12^
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp-sequential_8/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_11/Conv2D/ReadVariableOp,sequential_8/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp-sequential_8/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_12/Conv2D/ReadVariableOp,sequential_8/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp-sequential_8/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_13/Conv2D/ReadVariableOp,sequential_8/conv2d_13/Conv2D/ReadVariableOp2\
,sequential_8/dense_12/BiasAdd/ReadVariableOp,sequential_8/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_12/MatMul/ReadVariableOp+sequential_8/dense_12/MatMul/ReadVariableOp2\
,sequential_8/dense_13/BiasAdd/ReadVariableOp,sequential_8/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_13/MatMul/ReadVariableOp+sequential_8/dense_13/MatMul/ReadVariableOp2?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_26/ReadVariableOp2sequential_9/batch_normalization_26/ReadVariableOp2l
4sequential_9/batch_normalization_26/ReadVariableOp_14sequential_9/batch_normalization_26/ReadVariableOp_12?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_27/ReadVariableOp2sequential_9/batch_normalization_27/ReadVariableOp2l
4sequential_9/batch_normalization_27/ReadVariableOp_14sequential_9/batch_normalization_27/ReadVariableOp_12?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_28/ReadVariableOp2sequential_9/batch_normalization_28/ReadVariableOp2l
4sequential_9/batch_normalization_28/ReadVariableOp_14sequential_9/batch_normalization_28/ReadVariableOp_12r
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2\
,sequential_9/dense_14/BiasAdd/ReadVariableOp,sequential_9/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_14/MatMul/ReadVariableOp+sequential_9/dense_14/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
$__inference_signature_wrapper_311141
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_3083382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_43_layer_call_and_return_conditional_losses_310078

inputs
identityq
SigmoidSigmoidinputs*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_dense_12_layer_call_fn_313970

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3090142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309305

inputs
conv2d_11_309247
conv2d_11_309249!
batch_normalization_23_309252!
batch_normalization_23_309254!
batch_normalization_23_309256!
batch_normalization_23_309258
conv2d_12_309262
conv2d_12_309264!
batch_normalization_24_309267!
batch_normalization_24_309269!
batch_normalization_24_309271!
batch_normalization_24_309273
conv2d_13_309277
conv2d_13_309279!
batch_normalization_25_309282!
batch_normalization_25_309284!
batch_normalization_25_309286!
batch_normalization_25_309288
dense_12_309293
dense_12_309295
dense_13_309299
dense_13_309301
identity??.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_309247conv2d_11_309249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_3086642#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_23_309252batch_normalization_23_309254batch_normalization_23_309256batch_normalization_23_309258*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_30871720
.batch_normalization_23/StatefulPartitionedCall?
activation_35/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_35_layer_call_and_return_conditional_losses_3087582
activation_35/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_35/PartitionedCall:output:0conv2d_12_309262conv2d_12_309264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_3087762#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_309267batch_normalization_24_309269batch_normalization_24_309271batch_normalization_24_309273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_30882920
.batch_normalization_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_36_layer_call_and_return_conditional_losses_3088702
activation_36/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_13_309277conv2d_13_309279*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_3088882#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_309282batch_normalization_25_309284batch_normalization_25_309286batch_normalization_25_309288*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_30894120
.batch_normalization_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_37_layer_call_and_return_conditional_losses_3089822
activation_37/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3089962
flatten_4/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_309293dense_12_309295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3090142"
 dense_12/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_38_layer_call_and_return_conditional_losses_3090352
activation_38/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0dense_13_309299dense_13_309301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3090532"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_13_layer_call_fn_313802

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_3088882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_38_layer_call_and_return_conditional_losses_313975

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_24_layer_call_fn_313760

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3085042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_308717

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?&
?
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_309834

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_308699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_313479

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313526

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?B
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309131
input_5
conv2d_11_309073
conv2d_11_309075!
batch_normalization_23_309078!
batch_normalization_23_309080!
batch_normalization_23_309082!
batch_normalization_23_309084
conv2d_12_309088
conv2d_12_309090!
batch_normalization_24_309093!
batch_normalization_24_309095!
batch_normalization_24_309097!
batch_normalization_24_309099
conv2d_13_309103
conv2d_13_309105!
batch_normalization_25_309108!
batch_normalization_25_309110!
batch_normalization_25_309112!
batch_normalization_25_309114
dense_12_309119
dense_12_309121
dense_13_309125
dense_13_309127
identity??.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_11_309073conv2d_11_309075*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_3086642#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_23_309078batch_normalization_23_309080batch_normalization_23_309082batch_normalization_23_309084*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_30871720
.batch_normalization_23/StatefulPartitionedCall?
activation_35/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_35_layer_call_and_return_conditional_losses_3087582
activation_35/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_35/PartitionedCall:output:0conv2d_12_309088conv2d_12_309090*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_3087762#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_309093batch_normalization_24_309095batch_normalization_24_309097batch_normalization_24_309099*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_30882920
.batch_normalization_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_36_layer_call_and_return_conditional_losses_3088702
activation_36/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_13_309103conv2d_13_309105*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_3088882#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_309108batch_normalization_25_309110batch_normalization_25_309112batch_normalization_25_309114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_30894120
.batch_normalization_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_37_layer_call_and_return_conditional_losses_3089822
activation_37/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3089962
flatten_4/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_309119dense_12_309121*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3090142"
 dense_12/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_38_layer_call_and_return_conditional_losses_3090352
activation_38/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0dense_13_309125dense_13_309127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3090532"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_5
?
J
.__inference_activation_40_layer_call_fn_314121

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_40_layer_call_and_return_conditional_losses_3099542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_activation_41_layer_call_fn_314195

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_41_layer_call_and_return_conditional_losses_3100072
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313683

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_25_layer_call_fn_313917

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3089232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_313420

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3102142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314233

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_309682

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_308888

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
/__inference_auto_encoder_4_layer_call_fn_312421
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_3108542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
J
.__inference_activation_37_layer_call_fn_313940

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_37_layer_call_and_return_conditional_losses_3089822
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_308996

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_27_layer_call_fn_314185

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3096372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_309242
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3091952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_5
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_309858

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_308829

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_37_layer_call_and_return_conditional_losses_308982

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313822

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?A
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309070
input_5
conv2d_11_308675
conv2d_11_308677!
batch_normalization_23_308744!
batch_normalization_23_308746!
batch_normalization_23_308748!
batch_normalization_23_308750
conv2d_12_308787
conv2d_12_308789!
batch_normalization_24_308856!
batch_normalization_24_308858!
batch_normalization_24_308860!
batch_normalization_24_308862
conv2d_13_308899
conv2d_13_308901!
batch_normalization_25_308968!
batch_normalization_25_308970!
batch_normalization_25_308972!
batch_normalization_25_308974
dense_12_309025
dense_12_309027
dense_13_309064
dense_13_309066
identity??.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_11_308675conv2d_11_308677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_3086642#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_23_308744batch_normalization_23_308746batch_normalization_23_308748batch_normalization_23_308750*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_30869920
.batch_normalization_23/StatefulPartitionedCall?
activation_35/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_35_layer_call_and_return_conditional_losses_3087582
activation_35/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_35/PartitionedCall:output:0conv2d_12_308787conv2d_12_308789*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_3087762#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_308856batch_normalization_24_308858batch_normalization_24_308860batch_normalization_24_308862*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_30881120
.batch_normalization_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_36_layer_call_and_return_conditional_losses_3088702
activation_36/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_13_308899conv2d_13_308901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_3088882#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_308968batch_normalization_25_308970batch_normalization_25_308972batch_normalization_25_308974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_30892320
.batch_normalization_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_37_layer_call_and_return_conditional_losses_3089822
activation_37/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3089962
flatten_4/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_309025dense_12_309027*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3090142"
 dense_12/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_38_layer_call_and_return_conditional_losses_3090352
activation_38/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0dense_13_309064dense_13_309066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3090532"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_5
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_308639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_308400

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_309901

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
/__inference_auto_encoder_4_layer_call_fn_312328
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_3108542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
-__inference_sequential_9_layer_call_fn_313030
dense_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3102142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_14_input
?
e
I__inference_activation_43_layer_call_and_return_conditional_losses_314274

inputs
identityq
SigmoidSigmoidinputs*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_308941

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313747

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_26_layer_call_fn_314111

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3094892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_308923

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_13_layer_call_fn_313999

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3090532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?6
__inference__traced_save_314647
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop9
5savev2_conv2d_transpose_16_kernel_read_readvariableop7
3savev2_conv2d_transpose_16_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableop9
5savev2_conv2d_transpose_17_kernel_read_readvariableop7
3savev2_conv2d_transpose_17_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableop9
5savev2_conv2d_transpose_18_kernel_read_readvariableop7
3savev2_conv2d_transpose_18_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableop9
5savev2_conv2d_transpose_19_kernel_read_readvariableop7
3savev2_conv2d_transpose_19_bias_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_16_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_16_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_17_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_18_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_28_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_28_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_19_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_16_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_16_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_17_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_18_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_28_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_28_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_19_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_19_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?;
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?:
value?:B?:tB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop5savev2_conv2d_transpose_16_kernel_read_readvariableop3savev2_conv2d_transpose_16_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop5savev2_conv2d_transpose_17_kernel_read_readvariableop3savev2_conv2d_transpose_17_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop5savev2_conv2d_transpose_18_kernel_read_readvariableop3savev2_conv2d_transpose_18_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop5savev2_conv2d_transpose_19_kernel_read_readvariableop3savev2_conv2d_transpose_19_bias_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop>savev2_adam_batch_normalization_24_gamma_m_read_readvariableop=savev2_adam_batch_normalization_24_beta_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop>savev2_adam_batch_normalization_25_gamma_m_read_readvariableop=savev2_adam_batch_normalization_25_beta_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_16_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_16_bias_m_read_readvariableop>savev2_adam_batch_normalization_26_gamma_m_read_readvariableop=savev2_adam_batch_normalization_26_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_17_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_27_gamma_m_read_readvariableop=savev2_adam_batch_normalization_27_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_18_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_28_gamma_m_read_readvariableop=savev2_adam_batch_normalization_28_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_19_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_19_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop>savev2_adam_batch_normalization_24_gamma_v_read_readvariableop=savev2_adam_batch_normalization_24_beta_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop>savev2_adam_batch_normalization_25_gamma_v_read_readvariableop=savev2_adam_batch_normalization_25_beta_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_16_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_16_bias_v_read_readvariableop>savev2_adam_batch_normalization_26_gamma_v_read_readvariableop=savev2_adam_batch_normalization_26_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_17_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_27_gamma_v_read_readvariableop=savev2_adam_batch_normalization_27_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_18_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_28_gamma_v_read_readvariableop=savev2_adam_batch_normalization_28_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_19_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypesx
v2t	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : : : @:@:@:@:@?:?:?:?:
??:?:	?::	? :? :??:?:?:?:@?:@:@:@: @: : : : :: : :@:@:?:?:?:?:@:@: : : : : : : : : @:@:@:@:@?:?:?:?:
??:?:	?::	? :? :??:?:?:?:@?:@:@:@: @: : : : :: : : : : @:@:@:@:@?:?:?:?:
??:?:	?::	? :? :??:?:?:?:@?:@:@:@: @: : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	? :!

_output_shapes	
:? :.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:, (
&
_output_shapes
: @: !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
:: &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
:@: )

_output_shapes
:@:!*

_output_shapes	
:?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
: : 1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :,4(
&
_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:-<)
'
_output_shapes
:@?:!=

_output_shapes	
:?:!>

_output_shapes	
:?:!?

_output_shapes	
:?:&@"
 
_output_shapes
:
??:!A

_output_shapes	
:?:%B!

_output_shapes
:	?: C

_output_shapes
::%D!

_output_shapes
:	? :!E

_output_shapes	
:? :.F*
(
_output_shapes
:??:!G

_output_shapes	
:?:!H

_output_shapes	
:?:!I

_output_shapes	
:?:-J)
'
_output_shapes
:@?: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:,N(
&
_output_shapes
: @: O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :,R(
&
_output_shapes
: : S

_output_shapes
::,T(
&
_output_shapes
: : U

_output_shapes
: : V

_output_shapes
: : W

_output_shapes
: :,X(
&
_output_shapes
: @: Y

_output_shapes
:@: Z

_output_shapes
:@: [

_output_shapes
:@:-\)
'
_output_shapes
:@?:!]

_output_shapes	
:?:!^

_output_shapes	
:?:!_

_output_shapes	
:?:&`"
 
_output_shapes
:
??:!a

_output_shapes	
:?:%b!

_output_shapes
:	?: c

_output_shapes
::%d!

_output_shapes
:	? :!e

_output_shapes	
:? :.f*
(
_output_shapes
:??:!g

_output_shapes	
:?:!h

_output_shapes	
:?:!i

_output_shapes	
:?:-j)
'
_output_shapes
:@?: k

_output_shapes
:@: l

_output_shapes
:@: m

_output_shapes
:@:,n(
&
_output_shapes
: @: o

_output_shapes
: : p

_output_shapes
: : q

_output_shapes
: :,r(
&
_output_shapes
: : s

_output_shapes
::t

_output_shapes
: 
?
e
I__inference_activation_40_layer_call_and_return_conditional_losses_309954

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?-
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311595
input_19
5sequential_8_conv2d_11_conv2d_readvariableop_resource:
6sequential_8_conv2d_11_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_23_readvariableop_resourceA
=sequential_8_batch_normalization_23_readvariableop_1_resourceP
Lsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_12_conv2d_readvariableop_resource:
6sequential_8_conv2d_12_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_24_readvariableop_resourceA
=sequential_8_batch_normalization_24_readvariableop_1_resourceP
Lsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource9
5sequential_8_conv2d_13_conv2d_readvariableop_resource:
6sequential_8_conv2d_13_biasadd_readvariableop_resource?
;sequential_8_batch_normalization_25_readvariableop_resourceA
=sequential_8_batch_normalization_25_readvariableop_1_resourceP
Lsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceR
Nsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8
4sequential_8_dense_12_matmul_readvariableop_resource9
5sequential_8_dense_12_biasadd_readvariableop_resource8
4sequential_8_dense_13_matmul_readvariableop_resource9
5sequential_8_dense_13_biasadd_readvariableop_resource8
4sequential_9_dense_14_matmul_readvariableop_resource9
5sequential_9_dense_14_biasadd_readvariableop_resourceM
Isequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_26_readvariableop_resourceA
=sequential_9_batch_normalization_26_readvariableop_1_resourceP
Lsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_27_readvariableop_resourceA
=sequential_9_batch_normalization_27_readvariableop_1_resourceP
Lsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource?
;sequential_9_batch_normalization_28_readvariableop_resourceA
=sequential_9_batch_normalization_28_readvariableop_1_resourceP
Lsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceR
Nsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceM
Isequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resourceD
@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource
identity??Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_23/ReadVariableOp?4sequential_8/batch_normalization_23/ReadVariableOp_1?Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_24/ReadVariableOp?4sequential_8/batch_normalization_24/ReadVariableOp_1?Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?2sequential_8/batch_normalization_25/ReadVariableOp?4sequential_8/batch_normalization_25/ReadVariableOp_1?-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?,sequential_8/conv2d_11/Conv2D/ReadVariableOp?-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?,sequential_8/conv2d_12/Conv2D/ReadVariableOp?-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?,sequential_8/conv2d_13/Conv2D/ReadVariableOp?,sequential_8/dense_12/BiasAdd/ReadVariableOp?+sequential_8/dense_12/MatMul/ReadVariableOp?,sequential_8/dense_13/BiasAdd/ReadVariableOp?+sequential_8/dense_13/MatMul/ReadVariableOp?Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_26/ReadVariableOp?4sequential_9/batch_normalization_26/ReadVariableOp_1?Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_27/ReadVariableOp?4sequential_9/batch_normalization_27/ReadVariableOp_1?Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_28/ReadVariableOp?4sequential_9/batch_normalization_28/ReadVariableOp_1?7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?,sequential_9/dense_14/BiasAdd/ReadVariableOp?+sequential_9/dense_14/MatMul/ReadVariableOp?
,sequential_8/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_11/Conv2D/ReadVariableOp?
sequential_8/conv2d_11/Conv2DConv2Dinput_14sequential_8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_8/conv2d_11/Conv2D?
-sequential_8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp?
sequential_8/conv2d_11/BiasAddBiasAdd&sequential_8/conv2d_11/Conv2D:output:05sequential_8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2 
sequential_8/conv2d_11/BiasAdd?
2sequential_8/batch_normalization_23/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_8/batch_normalization_23/ReadVariableOp?
4sequential_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_8/batch_normalization_23/ReadVariableOp_1?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_11/BiasAdd:output:0:sequential_8/batch_normalization_23/ReadVariableOp:value:0<sequential_8/batch_normalization_23/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_23/FusedBatchNormV3?
sequential_8/activation_35/ReluRelu8sequential_8/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2!
sequential_8/activation_35/Relu?
,sequential_8/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_8/conv2d_12/Conv2D/ReadVariableOp?
sequential_8/conv2d_12/Conv2DConv2D-sequential_8/activation_35/Relu:activations:04sequential_8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_8/conv2d_12/Conv2D?
-sequential_8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp?
sequential_8/conv2d_12/BiasAddBiasAdd&sequential_8/conv2d_12/Conv2D:output:05sequential_8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_8/conv2d_12/BiasAdd?
2sequential_8/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_8/batch_normalization_24/ReadVariableOp?
4sequential_8/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_8/batch_normalization_24/ReadVariableOp_1?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_12/BiasAdd:output:0:sequential_8/batch_normalization_24/ReadVariableOp:value:0<sequential_8/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_24/FusedBatchNormV3?
sequential_8/activation_36/ReluRelu8sequential_8/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_8/activation_36/Relu?
,sequential_8/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02.
,sequential_8/conv2d_13/Conv2D/ReadVariableOp?
sequential_8/conv2d_13/Conv2DConv2D-sequential_8/activation_36/Relu:activations:04sequential_8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
sequential_8/conv2d_13/Conv2D?
-sequential_8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp?
sequential_8/conv2d_13/BiasAddBiasAdd&sequential_8/conv2d_13/Conv2D:output:05sequential_8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_8/conv2d_13/BiasAdd?
2sequential_8/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_8/batch_normalization_25/ReadVariableOp?
4sequential_8/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_8/batch_normalization_25/ReadVariableOp_1?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
4sequential_8/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3'sequential_8/conv2d_13/BiasAdd:output:0:sequential_8/batch_normalization_25/ReadVariableOp:value:0<sequential_8/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_8/batch_normalization_25/FusedBatchNormV3?
sequential_8/activation_37/ReluRelu8sequential_8/batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_8/activation_37/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape-sequential_8/activation_37/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
+sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_8/dense_12/MatMul/ReadVariableOp?
sequential_8/dense_12/MatMulMatMul'sequential_8/flatten_4/Reshape:output:03sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/MatMul?
,sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_8/dense_12/BiasAdd/ReadVariableOp?
sequential_8/dense_12/BiasAddBiasAdd&sequential_8/dense_12/MatMul:product:04sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_8/dense_12/BiasAdd?
sequential_8/activation_38/ReluRelu&sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_8/activation_38/Relu?
+sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_8/dense_13/MatMul/ReadVariableOp?
sequential_8/dense_13/MatMulMatMul-sequential_8/activation_38/Relu:activations:03sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/MatMul?
,sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_8/dense_13/BiasAdd/ReadVariableOp?
sequential_8/dense_13/BiasAddBiasAdd&sequential_8/dense_13/MatMul:product:04sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_13/BiasAdd?
+sequential_9/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02-
+sequential_9/dense_14/MatMul/ReadVariableOp?
sequential_9/dense_14/MatMulMatMul&sequential_8/dense_13/BiasAdd:output:03sequential_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/MatMul?
,sequential_9/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02.
,sequential_9/dense_14/BiasAdd/ReadVariableOp?
sequential_9/dense_14/BiasAddBiasAdd&sequential_9/dense_14/MatMul:product:04sequential_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
sequential_9/dense_14/BiasAdd?
sequential_9/activation_39/ReluRelu&sequential_9/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2!
sequential_9/activation_39/Relu?
sequential_9/reshape_4/ShapeShape-sequential_9/activation_39/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_4/Shape?
*sequential_9/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/reshape_4/strided_slice/stack?
,sequential_9/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_1?
,sequential_9/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/reshape_4/strided_slice/stack_2?
$sequential_9/reshape_4/strided_sliceStridedSlice%sequential_9/reshape_4/Shape:output:03sequential_9/reshape_4/strided_slice/stack:output:05sequential_9/reshape_4/strided_slice/stack_1:output:05sequential_9/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/reshape_4/strided_slice?
&sequential_9/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/1?
&sequential_9/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_9/reshape_4/Reshape/shape/2?
&sequential_9/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_9/reshape_4/Reshape/shape/3?
$sequential_9/reshape_4/Reshape/shapePack-sequential_9/reshape_4/strided_slice:output:0/sequential_9/reshape_4/Reshape/shape/1:output:0/sequential_9/reshape_4/Reshape/shape/2:output:0/sequential_9/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/reshape_4/Reshape/shape?
sequential_9/reshape_4/ReshapeReshape-sequential_9/activation_39/Relu:activations:0-sequential_9/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2 
sequential_9/reshape_4/Reshape?
&sequential_9/conv2d_transpose_16/ShapeShape'sequential_9/reshape_4/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/Shape?
4sequential_9/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_16/strided_slice/stack?
6sequential_9/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_1?
6sequential_9/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_16/strided_slice/stack_2?
.sequential_9/conv2d_transpose_16/strided_sliceStridedSlice/sequential_9/conv2d_transpose_16/Shape:output:0=sequential_9/conv2d_transpose_16/strided_slice/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_16/strided_slice?
(sequential_9/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/1?
(sequential_9/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_16/stack/2?
(sequential_9/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_9/conv2d_transpose_16/stack/3?
&sequential_9/conv2d_transpose_16/stackPack7sequential_9/conv2d_transpose_16/strided_slice:output:01sequential_9/conv2d_transpose_16/stack/1:output:01sequential_9/conv2d_transpose_16/stack/2:output:01sequential_9/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_16/stack?
6sequential_9/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_16/strided_slice_1/stack?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_16/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_16/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_16/stack:output:0?sequential_9/conv2d_transpose_16/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_16/strided_slice_1?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_16/stack:output:0Hsequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0'sequential_9/reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_16/conv2d_transpose?
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_16/BiasAddBiasAdd:sequential_9/conv2d_transpose_16/conv2d_transpose:output:0?sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2*
(sequential_9/conv2d_transpose_16/BiasAdd?
2sequential_9/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_9/batch_normalization_26/ReadVariableOp?
4sequential_9/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_9/batch_normalization_26/ReadVariableOp_1?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_16/BiasAdd:output:0:sequential_9/batch_normalization_26/ReadVariableOp:value:0<sequential_9/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_26/FusedBatchNormV3?
sequential_9/activation_40/ReluRelu8sequential_9/batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2!
sequential_9/activation_40/Relu?
&sequential_9/conv2d_transpose_17/ShapeShape-sequential_9/activation_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/Shape?
4sequential_9/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_17/strided_slice/stack?
6sequential_9/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_1?
6sequential_9/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_17/strided_slice/stack_2?
.sequential_9/conv2d_transpose_17/strided_sliceStridedSlice/sequential_9/conv2d_transpose_17/Shape:output:0=sequential_9/conv2d_transpose_17/strided_slice/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_17/strided_slice?
(sequential_9/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/1?
(sequential_9/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_17/stack/2?
(sequential_9/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_17/stack/3?
&sequential_9/conv2d_transpose_17/stackPack7sequential_9/conv2d_transpose_17/strided_slice:output:01sequential_9/conv2d_transpose_17/stack/1:output:01sequential_9/conv2d_transpose_17/stack/2:output:01sequential_9/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_17/stack?
6sequential_9/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_17/strided_slice_1/stack?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_17/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_17/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_17/stack:output:0?sequential_9/conv2d_transpose_17/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_17/strided_slice_1?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_17/stack:output:0Hsequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_17/conv2d_transpose?
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_17/BiasAddBiasAdd:sequential_9/conv2d_transpose_17/conv2d_transpose:output:0?sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_17/BiasAdd?
2sequential_9/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_9/batch_normalization_27/ReadVariableOp?
4sequential_9/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_9/batch_normalization_27/ReadVariableOp_1?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_27/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_17/BiasAdd:output:0:sequential_9/batch_normalization_27/ReadVariableOp:value:0<sequential_9/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_27/FusedBatchNormV3?
sequential_9/activation_41/ReluRelu8sequential_9/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2!
sequential_9/activation_41/Relu?
&sequential_9/conv2d_transpose_18/ShapeShape-sequential_9/activation_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/Shape?
4sequential_9/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_18/strided_slice/stack?
6sequential_9/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_1?
6sequential_9/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_18/strided_slice/stack_2?
.sequential_9/conv2d_transpose_18/strided_sliceStridedSlice/sequential_9/conv2d_transpose_18/Shape:output:0=sequential_9/conv2d_transpose_18/strided_slice/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_18/strided_slice?
(sequential_9/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/1?
(sequential_9/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/2?
(sequential_9/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_18/stack/3?
&sequential_9/conv2d_transpose_18/stackPack7sequential_9/conv2d_transpose_18/strided_slice:output:01sequential_9/conv2d_transpose_18/stack/1:output:01sequential_9/conv2d_transpose_18/stack/2:output:01sequential_9/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_18/stack?
6sequential_9/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_18/strided_slice_1/stack?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_18/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_18/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_18/stack:output:0?sequential_9/conv2d_transpose_18/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_18/strided_slice_1?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_18/stack:output:0Hsequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_18/conv2d_transpose?
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_18/BiasAddBiasAdd:sequential_9/conv2d_transpose_18/conv2d_transpose:output:0?sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2*
(sequential_9/conv2d_transpose_18/BiasAdd?
2sequential_9/batch_normalization_28/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_9/batch_normalization_28/ReadVariableOp?
4sequential_9/batch_normalization_28/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_9/batch_normalization_28/ReadVariableOp_1?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
4sequential_9/batch_normalization_28/FusedBatchNormV3FusedBatchNormV31sequential_9/conv2d_transpose_18/BiasAdd:output:0:sequential_9/batch_normalization_28/ReadVariableOp:value:0<sequential_9/batch_normalization_28/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 26
4sequential_9/batch_normalization_28/FusedBatchNormV3?
sequential_9/activation_42/ReluRelu8sequential_9/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2!
sequential_9/activation_42/Relu?
&sequential_9/conv2d_transpose_19/ShapeShape-sequential_9/activation_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/Shape?
4sequential_9/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_19/strided_slice/stack?
6sequential_9/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_1?
6sequential_9/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_19/strided_slice/stack_2?
.sequential_9/conv2d_transpose_19/strided_sliceStridedSlice/sequential_9/conv2d_transpose_19/Shape:output:0=sequential_9/conv2d_transpose_19/strided_slice/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_19/strided_slice?
(sequential_9/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/1?
(sequential_9/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_19/stack/2?
(sequential_9/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_19/stack/3?
&sequential_9/conv2d_transpose_19/stackPack7sequential_9/conv2d_transpose_19/strided_slice:output:01sequential_9/conv2d_transpose_19/stack/1:output:01sequential_9/conv2d_transpose_19/stack/2:output:01sequential_9/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_19/stack?
6sequential_9/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_19/strided_slice_1/stack?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_19/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_19/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_19/stack:output:0?sequential_9/conv2d_transpose_19/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_19/strided_slice_1?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_19/stack:output:0Hsequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0-sequential_9/activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
23
1sequential_9/conv2d_transpose_19/conv2d_transpose?
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_19/BiasAddBiasAdd:sequential_9/conv2d_transpose_19/conv2d_transpose:output:0?sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2*
(sequential_9/conv2d_transpose_19/BiasAdd?
"sequential_9/activation_43/SigmoidSigmoid1sequential_9/conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"sequential_9/activation_43/Sigmoid?
IdentityIdentity&sequential_9/activation_43/Sigmoid:y:0D^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_23/ReadVariableOp5^sequential_8/batch_normalization_23/ReadVariableOp_1D^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_24/ReadVariableOp5^sequential_8/batch_normalization_24/ReadVariableOp_1D^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_25/ReadVariableOp5^sequential_8/batch_normalization_25/ReadVariableOp_1.^sequential_8/conv2d_11/BiasAdd/ReadVariableOp-^sequential_8/conv2d_11/Conv2D/ReadVariableOp.^sequential_8/conv2d_12/BiasAdd/ReadVariableOp-^sequential_8/conv2d_12/Conv2D/ReadVariableOp.^sequential_8/conv2d_13/BiasAdd/ReadVariableOp-^sequential_8/conv2d_13/Conv2D/ReadVariableOp-^sequential_8/dense_12/BiasAdd/ReadVariableOp,^sequential_8/dense_12/MatMul/ReadVariableOp-^sequential_8/dense_13/BiasAdd/ReadVariableOp,^sequential_8/dense_13/MatMul/ReadVariableOpD^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_26/ReadVariableOp5^sequential_9/batch_normalization_26/ReadVariableOp_1D^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_27/ReadVariableOp5^sequential_9/batch_normalization_27/ReadVariableOp_1D^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_28/ReadVariableOp5^sequential_9/batch_normalization_28/ReadVariableOp_18^sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp-^sequential_9/dense_14/BiasAdd/ReadVariableOp,^sequential_9/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2?
Csequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_23/ReadVariableOp2sequential_8/batch_normalization_23/ReadVariableOp2l
4sequential_8/batch_normalization_23/ReadVariableOp_14sequential_8/batch_normalization_23/ReadVariableOp_12?
Csequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_24/ReadVariableOp2sequential_8/batch_normalization_24/ReadVariableOp2l
4sequential_8/batch_normalization_24/ReadVariableOp_14sequential_8/batch_normalization_24/ReadVariableOp_12?
Csequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_25/ReadVariableOp2sequential_8/batch_normalization_25/ReadVariableOp2l
4sequential_8/batch_normalization_25/ReadVariableOp_14sequential_8/batch_normalization_25/ReadVariableOp_12^
-sequential_8/conv2d_11/BiasAdd/ReadVariableOp-sequential_8/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_11/Conv2D/ReadVariableOp,sequential_8/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_12/BiasAdd/ReadVariableOp-sequential_8/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_12/Conv2D/ReadVariableOp,sequential_8/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_13/BiasAdd/ReadVariableOp-sequential_8/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_13/Conv2D/ReadVariableOp,sequential_8/conv2d_13/Conv2D/ReadVariableOp2\
,sequential_8/dense_12/BiasAdd/ReadVariableOp,sequential_8/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_12/MatMul/ReadVariableOp+sequential_8/dense_12/MatMul/ReadVariableOp2\
,sequential_8/dense_13/BiasAdd/ReadVariableOp,sequential_8/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_13/MatMul/ReadVariableOp+sequential_8/dense_13/MatMul/ReadVariableOp2?
Csequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_26/ReadVariableOp2sequential_9/batch_normalization_26/ReadVariableOp2l
4sequential_9/batch_normalization_26/ReadVariableOp_14sequential_9/batch_normalization_26/ReadVariableOp_12?
Csequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_27/ReadVariableOp2sequential_9/batch_normalization_27/ReadVariableOp2l
4sequential_9/batch_normalization_27/ReadVariableOp_14sequential_9/batch_normalization_27/ReadVariableOp_12?
Csequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_28/ReadVariableOp2sequential_9/batch_normalization_28/ReadVariableOp2l
4sequential_9/batch_normalization_28/ReadVariableOp_14sequential_9/batch_normalization_28/ReadVariableOp_12r
7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2\
,sequential_9/dense_14/BiasAdd/ReadVariableOp,sequential_9/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_14/MatMul/ReadVariableOp+sequential_9/dense_14/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
J
.__inference_activation_38_layer_call_fn_313980

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_38_layer_call_and_return_conditional_losses_3090352
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_28_layer_call_fn_314259

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_3097852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_308608

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_313946

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_35_layer_call_and_return_conditional_losses_313621

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_42_layer_call_and_return_conditional_losses_314264

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?7
!__inference__wrapped_model_308338
input_1H
Dauto_encoder_4_sequential_8_conv2d_11_conv2d_readvariableop_resourceI
Eauto_encoder_4_sequential_8_conv2d_11_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_8_batch_normalization_23_readvariableop_resourceP
Lauto_encoder_4_sequential_8_batch_normalization_23_readvariableop_1_resource_
[auto_encoder_4_sequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceH
Dauto_encoder_4_sequential_8_conv2d_12_conv2d_readvariableop_resourceI
Eauto_encoder_4_sequential_8_conv2d_12_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_8_batch_normalization_24_readvariableop_resourceP
Lauto_encoder_4_sequential_8_batch_normalization_24_readvariableop_1_resource_
[auto_encoder_4_sequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceH
Dauto_encoder_4_sequential_8_conv2d_13_conv2d_readvariableop_resourceI
Eauto_encoder_4_sequential_8_conv2d_13_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_8_batch_normalization_25_readvariableop_resourceP
Lauto_encoder_4_sequential_8_batch_normalization_25_readvariableop_1_resource_
[auto_encoder_4_sequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceG
Cauto_encoder_4_sequential_8_dense_12_matmul_readvariableop_resourceH
Dauto_encoder_4_sequential_8_dense_12_biasadd_readvariableop_resourceG
Cauto_encoder_4_sequential_8_dense_13_matmul_readvariableop_resourceH
Dauto_encoder_4_sequential_8_dense_13_biasadd_readvariableop_resourceG
Cauto_encoder_4_sequential_9_dense_14_matmul_readvariableop_resourceH
Dauto_encoder_4_sequential_9_dense_14_biasadd_readvariableop_resource\
Xauto_encoder_4_sequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resourceS
Oauto_encoder_4_sequential_9_conv2d_transpose_16_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_9_batch_normalization_26_readvariableop_resourceP
Lauto_encoder_4_sequential_9_batch_normalization_26_readvariableop_1_resource_
[auto_encoder_4_sequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource\
Xauto_encoder_4_sequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resourceS
Oauto_encoder_4_sequential_9_conv2d_transpose_17_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_9_batch_normalization_27_readvariableop_resourceP
Lauto_encoder_4_sequential_9_batch_normalization_27_readvariableop_1_resource_
[auto_encoder_4_sequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource\
Xauto_encoder_4_sequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resourceS
Oauto_encoder_4_sequential_9_conv2d_transpose_18_biasadd_readvariableop_resourceN
Jauto_encoder_4_sequential_9_batch_normalization_28_readvariableop_resourceP
Lauto_encoder_4_sequential_9_batch_normalization_28_readvariableop_1_resource_
[auto_encoder_4_sequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resourcea
]auto_encoder_4_sequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource\
Xauto_encoder_4_sequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resourceS
Oauto_encoder_4_sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource
identity??Rauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp?Cauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1?Rauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp?Cauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1?Rauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp?Cauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1?<auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp?;auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp?<auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp?;auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp?<auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp?;auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp?;auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp?:auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp?;auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp?:auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOp?Rauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp?Cauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1?Rauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp?Cauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1?Rauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Tauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?Aauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp?Cauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1?Fauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?Oauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?Fauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?Oauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?Fauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?Oauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?Fauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?Oauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?;auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp?:auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp?
;auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp?
,auto_encoder_4/sequential_8/conv2d_11/Conv2DConv2Dinput_1Cauto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2.
,auto_encoder_4/sequential_8/conv2d_11/Conv2D?
<auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpEauto_encoder_4_sequential_8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp?
-auto_encoder_4/sequential_8/conv2d_11/BiasAddBiasAdd5auto_encoder_4/sequential_8/conv2d_11/Conv2D:output:0Dauto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2/
-auto_encoder_4/sequential_8/conv2d_11/BiasAdd?
Aauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_8_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype02C
Aauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp?
Cauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1?
Rauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02T
Rauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02V
Tauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV36auto_encoder_4/sequential_8/conv2d_11/BiasAdd:output:0Iauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp:value:0Kauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3?
.auto_encoder_4/sequential_8/activation_35/ReluReluGauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 20
.auto_encoder_4/sequential_8/activation_35/Relu?
;auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp?
,auto_encoder_4/sequential_8/conv2d_12/Conv2DConv2D<auto_encoder_4/sequential_8/activation_35/Relu:activations:0Cauto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2.
,auto_encoder_4/sequential_8/conv2d_12/Conv2D?
<auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpEauto_encoder_4_sequential_8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp?
-auto_encoder_4/sequential_8/conv2d_12/BiasAddBiasAdd5auto_encoder_4/sequential_8/conv2d_12/Conv2D:output:0Dauto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2/
-auto_encoder_4/sequential_8/conv2d_12/BiasAdd?
Aauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_8_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp?
Cauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_8_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1?
Rauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_8_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02V
Tauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3FusedBatchNormV36auto_encoder_4/sequential_8/conv2d_12/BiasAdd:output:0Iauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp:value:0Kauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3?
.auto_encoder_4/sequential_8/activation_36/ReluReluGauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@20
.auto_encoder_4/sequential_8/activation_36/Relu?
;auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_8_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02=
;auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp?
,auto_encoder_4/sequential_8/conv2d_13/Conv2DConv2D<auto_encoder_4/sequential_8/activation_36/Relu:activations:0Cauto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2.
,auto_encoder_4/sequential_8/conv2d_13/Conv2D?
<auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpEauto_encoder_4_sequential_8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp?
-auto_encoder_4/sequential_8/conv2d_13/BiasAddBiasAdd5auto_encoder_4/sequential_8/conv2d_13/Conv2D:output:0Dauto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2/
-auto_encoder_4/sequential_8/conv2d_13/BiasAdd?
Aauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_8_batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Aauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp?
Cauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_8_batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Cauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1?
Rauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_8_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02V
Tauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3FusedBatchNormV36auto_encoder_4/sequential_8/conv2d_13/BiasAdd:output:0Iauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp:value:0Kauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3?
.auto_encoder_4/sequential_8/activation_37/ReluReluGauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????20
.auto_encoder_4/sequential_8/activation_37/Relu?
+auto_encoder_4/sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2-
+auto_encoder_4/sequential_8/flatten_4/Const?
-auto_encoder_4/sequential_8/flatten_4/ReshapeReshape<auto_encoder_4/sequential_8/activation_37/Relu:activations:04auto_encoder_4/sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2/
-auto_encoder_4/sequential_8/flatten_4/Reshape?
:auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOpCauto_encoder_4_sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp?
+auto_encoder_4/sequential_8/dense_12/MatMulMatMul6auto_encoder_4/sequential_8/flatten_4/Reshape:output:0Bauto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+auto_encoder_4/sequential_8/dense_12/MatMul?
;auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp?
,auto_encoder_4/sequential_8/dense_12/BiasAddBiasAdd5auto_encoder_4/sequential_8/dense_12/MatMul:product:0Cauto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,auto_encoder_4/sequential_8/dense_12/BiasAdd?
.auto_encoder_4/sequential_8/activation_38/ReluRelu5auto_encoder_4/sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????20
.auto_encoder_4/sequential_8/activation_38/Relu?
:auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOpCauto_encoder_4_sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02<
:auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOp?
+auto_encoder_4/sequential_8/dense_13/MatMulMatMul<auto_encoder_4/sequential_8/activation_38/Relu:activations:0Bauto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+auto_encoder_4/sequential_8/dense_13/MatMul?
;auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp?
,auto_encoder_4/sequential_8/dense_13/BiasAddBiasAdd5auto_encoder_4/sequential_8/dense_13/MatMul:product:0Cauto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,auto_encoder_4/sequential_8/dense_13/BiasAdd?
:auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOpReadVariableOpCauto_encoder_4_sequential_9_dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02<
:auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp?
+auto_encoder_4/sequential_9/dense_14/MatMulMatMul5auto_encoder_4/sequential_8/dense_13/BiasAdd:output:0Bauto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2-
+auto_encoder_4/sequential_9/dense_14/MatMul?
;auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOpReadVariableOpDauto_encoder_4_sequential_9_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02=
;auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp?
,auto_encoder_4/sequential_9/dense_14/BiasAddBiasAdd5auto_encoder_4/sequential_9/dense_14/MatMul:product:0Cauto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2.
,auto_encoder_4/sequential_9/dense_14/BiasAdd?
.auto_encoder_4/sequential_9/activation_39/ReluRelu5auto_encoder_4/sequential_9/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 20
.auto_encoder_4/sequential_9/activation_39/Relu?
+auto_encoder_4/sequential_9/reshape_4/ShapeShape<auto_encoder_4/sequential_9/activation_39/Relu:activations:0*
T0*
_output_shapes
:2-
+auto_encoder_4/sequential_9/reshape_4/Shape?
9auto_encoder_4/sequential_9/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9auto_encoder_4/sequential_9/reshape_4/strided_slice/stack?
;auto_encoder_4/sequential_9/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;auto_encoder_4/sequential_9/reshape_4/strided_slice/stack_1?
;auto_encoder_4/sequential_9/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;auto_encoder_4/sequential_9/reshape_4/strided_slice/stack_2?
3auto_encoder_4/sequential_9/reshape_4/strided_sliceStridedSlice4auto_encoder_4/sequential_9/reshape_4/Shape:output:0Bauto_encoder_4/sequential_9/reshape_4/strided_slice/stack:output:0Dauto_encoder_4/sequential_9/reshape_4/strided_slice/stack_1:output:0Dauto_encoder_4/sequential_9/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3auto_encoder_4/sequential_9/reshape_4/strided_slice?
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/1?
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/2?
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?27
5auto_encoder_4/sequential_9/reshape_4/Reshape/shape/3?
3auto_encoder_4/sequential_9/reshape_4/Reshape/shapePack<auto_encoder_4/sequential_9/reshape_4/strided_slice:output:0>auto_encoder_4/sequential_9/reshape_4/Reshape/shape/1:output:0>auto_encoder_4/sequential_9/reshape_4/Reshape/shape/2:output:0>auto_encoder_4/sequential_9/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3auto_encoder_4/sequential_9/reshape_4/Reshape/shape?
-auto_encoder_4/sequential_9/reshape_4/ReshapeReshape<auto_encoder_4/sequential_9/activation_39/Relu:activations:0<auto_encoder_4/sequential_9/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2/
-auto_encoder_4/sequential_9/reshape_4/Reshape?
5auto_encoder_4/sequential_9/conv2d_transpose_16/ShapeShape6auto_encoder_4/sequential_9/reshape_4/Reshape:output:0*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_16/Shape?
Cauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_1?
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_2?
=auto_encoder_4/sequential_9/conv2d_transpose_16/strided_sliceStridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_16/Shape:output:0Lauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_1:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=auto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice?
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :29
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/1?
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :29
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/2?
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?29
7auto_encoder_4/sequential_9/conv2d_transpose_16/stack/3?
5auto_encoder_4/sequential_9/conv2d_transpose_16/stackPackFauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice:output:0@auto_encoder_4/sequential_9/conv2d_transpose_16/stack/1:output:0@auto_encoder_4/sequential_9/conv2d_transpose_16/stack/2:output:0@auto_encoder_4/sequential_9/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_16/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Eauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack?
Gauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_1?
Gauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_2?
?auto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1StridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_16/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_1:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?auto_encoder_4/sequential_9/conv2d_transpose_16/strided_slice_1?
Oauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpXauto_encoder_4_sequential_9_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02Q
Oauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
@auto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput>auto_encoder_4/sequential_9/conv2d_transpose_16/stack:output:0Wauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:06auto_encoder_4/sequential_9/reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2B
@auto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose?
Fauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOpOauto_encoder_4_sequential_9_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp?
7auto_encoder_4/sequential_9/conv2d_transpose_16/BiasAddBiasAddIauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????29
7auto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd?
Aauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_9_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Aauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp?
Cauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_9_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Cauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1?
Rauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_9_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02V
Tauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3@auto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd:output:0Iauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp:value:0Kauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3?
.auto_encoder_4/sequential_9/activation_40/ReluReluGauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????20
.auto_encoder_4/sequential_9/activation_40/Relu?
5auto_encoder_4/sequential_9/conv2d_transpose_17/ShapeShape<auto_encoder_4/sequential_9/activation_40/Relu:activations:0*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_17/Shape?
Cauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_1?
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_2?
=auto_encoder_4/sequential_9/conv2d_transpose_17/strided_sliceStridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_17/Shape:output:0Lauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_1:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=auto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice?
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :29
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/1?
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :29
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/2?
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@29
7auto_encoder_4/sequential_9/conv2d_transpose_17/stack/3?
5auto_encoder_4/sequential_9/conv2d_transpose_17/stackPackFauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice:output:0@auto_encoder_4/sequential_9/conv2d_transpose_17/stack/1:output:0@auto_encoder_4/sequential_9/conv2d_transpose_17/stack/2:output:0@auto_encoder_4/sequential_9/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_17/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Eauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack?
Gauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_1?
Gauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_2?
?auto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1StridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_17/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_1:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?auto_encoder_4/sequential_9/conv2d_transpose_17/strided_slice_1?
Oauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpXauto_encoder_4_sequential_9_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02Q
Oauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
@auto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput>auto_encoder_4/sequential_9/conv2d_transpose_17/stack:output:0Wauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0<auto_encoder_4/sequential_9/activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2B
@auto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose?
Fauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOpOauto_encoder_4_sequential_9_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp?
7auto_encoder_4/sequential_9/conv2d_transpose_17/BiasAddBiasAddIauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@29
7auto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd?
Aauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_9_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp?
Cauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_9_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1?
Rauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_9_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02V
Tauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3@auto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd:output:0Iauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp:value:0Kauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3?
.auto_encoder_4/sequential_9/activation_41/ReluReluGauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@20
.auto_encoder_4/sequential_9/activation_41/Relu?
5auto_encoder_4/sequential_9/conv2d_transpose_18/ShapeShape<auto_encoder_4/sequential_9/activation_41/Relu:activations:0*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_18/Shape?
Cauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_1?
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_2?
=auto_encoder_4/sequential_9/conv2d_transpose_18/strided_sliceStridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_18/Shape:output:0Lauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_1:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=auto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice?
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 29
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/1?
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 29
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/2?
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 29
7auto_encoder_4/sequential_9/conv2d_transpose_18/stack/3?
5auto_encoder_4/sequential_9/conv2d_transpose_18/stackPackFauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice:output:0@auto_encoder_4/sequential_9/conv2d_transpose_18/stack/1:output:0@auto_encoder_4/sequential_9/conv2d_transpose_18/stack/2:output:0@auto_encoder_4/sequential_9/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_18/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Eauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack?
Gauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_1?
Gauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_2?
?auto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1StridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_18/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_1:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?auto_encoder_4/sequential_9/conv2d_transpose_18/strided_slice_1?
Oauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpXauto_encoder_4_sequential_9_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02Q
Oauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
@auto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput>auto_encoder_4/sequential_9/conv2d_transpose_18/stack:output:0Wauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0<auto_encoder_4/sequential_9/activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2B
@auto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose?
Fauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOpOauto_encoder_4_sequential_9_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02H
Fauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp?
7auto_encoder_4/sequential_9/conv2d_transpose_18/BiasAddBiasAddIauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   29
7auto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd?
Aauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOpReadVariableOpJauto_encoder_4_sequential_9_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02C
Aauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp?
Cauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1ReadVariableOpLauto_encoder_4_sequential_9_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1?
Rauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp[auto_encoder_4_sequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02T
Rauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Tauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]auto_encoder_4_sequential_9_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02V
Tauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
Cauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3@auto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd:output:0Iauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp:value:0Kauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1:value:0Zauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0\auto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2E
Cauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3?
.auto_encoder_4/sequential_9/activation_42/ReluReluGauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   20
.auto_encoder_4/sequential_9/activation_42/Relu?
5auto_encoder_4/sequential_9/conv2d_transpose_19/ShapeShape<auto_encoder_4/sequential_9/activation_42/Relu:activations:0*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_19/Shape?
Cauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_1?
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_2?
=auto_encoder_4/sequential_9/conv2d_transpose_19/strided_sliceStridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_19/Shape:output:0Lauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_1:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=auto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice?
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 29
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/1?
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 29
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/2?
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :29
7auto_encoder_4/sequential_9/conv2d_transpose_19/stack/3?
5auto_encoder_4/sequential_9/conv2d_transpose_19/stackPackFauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice:output:0@auto_encoder_4/sequential_9/conv2d_transpose_19/stack/1:output:0@auto_encoder_4/sequential_9/conv2d_transpose_19/stack/2:output:0@auto_encoder_4/sequential_9/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:27
5auto_encoder_4/sequential_9/conv2d_transpose_19/stack?
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Eauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack?
Gauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_1?
Gauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_2?
?auto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1StridedSlice>auto_encoder_4/sequential_9/conv2d_transpose_19/stack:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_1:output:0Pauto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?auto_encoder_4/sequential_9/conv2d_transpose_19/strided_slice_1?
Oauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpXauto_encoder_4_sequential_9_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02Q
Oauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
@auto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput>auto_encoder_4/sequential_9/conv2d_transpose_19/stack:output:0Wauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0<auto_encoder_4/sequential_9/activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2B
@auto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose?
Fauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOpOauto_encoder_4_sequential_9_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02H
Fauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp?
7auto_encoder_4/sequential_9/conv2d_transpose_19/BiasAddBiasAddIauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose:output:0Nauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  29
7auto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd?
1auto_encoder_4/sequential_9/activation_43/SigmoidSigmoid@auto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  23
1auto_encoder_4/sequential_9/activation_43/Sigmoid?
IdentityIdentity5auto_encoder_4/sequential_9/activation_43/Sigmoid:y:0S^auto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOpD^auto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1S^auto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOpD^auto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1S^auto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOpD^auto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1=^auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp<^auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp=^auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp<^auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp=^auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp<^auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp<^auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp;^auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp<^auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp;^auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOpS^auto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOpD^auto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1S^auto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOpD^auto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1S^auto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpU^auto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1B^auto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOpD^auto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1G^auto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpP^auto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpG^auto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpP^auto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpG^auto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpP^auto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpG^auto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpP^auto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp<^auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp;^auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2?
Rauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOpAauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp2?
Cauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_1Cauto_encoder_4/sequential_8/batch_normalization_23/ReadVariableOp_12?
Rauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_8/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOpAauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp2?
Cauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_1Cauto_encoder_4/sequential_8/batch_normalization_24/ReadVariableOp_12?
Rauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_8/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOpAauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp2?
Cauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_1Cauto_encoder_4/sequential_8/batch_normalization_25/ReadVariableOp_12|
<auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp<auto_encoder_4/sequential_8/conv2d_11/BiasAdd/ReadVariableOp2z
;auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp;auto_encoder_4/sequential_8/conv2d_11/Conv2D/ReadVariableOp2|
<auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp<auto_encoder_4/sequential_8/conv2d_12/BiasAdd/ReadVariableOp2z
;auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp;auto_encoder_4/sequential_8/conv2d_12/Conv2D/ReadVariableOp2|
<auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp<auto_encoder_4/sequential_8/conv2d_13/BiasAdd/ReadVariableOp2z
;auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp;auto_encoder_4/sequential_8/conv2d_13/Conv2D/ReadVariableOp2z
;auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp;auto_encoder_4/sequential_8/dense_12/BiasAdd/ReadVariableOp2x
:auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp:auto_encoder_4/sequential_8/dense_12/MatMul/ReadVariableOp2z
;auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp;auto_encoder_4/sequential_8/dense_13/BiasAdd/ReadVariableOp2x
:auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOp:auto_encoder_4/sequential_8/dense_13/MatMul/ReadVariableOp2?
Rauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_9/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOpAauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp2?
Cauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_1Cauto_encoder_4/sequential_9/batch_normalization_26/ReadVariableOp_12?
Rauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_9/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOpAauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp2?
Cauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_1Cauto_encoder_4/sequential_9/batch_normalization_27/ReadVariableOp_12?
Rauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOpRauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Tauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Tauto_encoder_4/sequential_9/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12?
Aauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOpAauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp2?
Cauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_1Cauto_encoder_4/sequential_9/batch_normalization_28/ReadVariableOp_12?
Fauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOpFauto_encoder_4/sequential_9/conv2d_transpose_16/BiasAdd/ReadVariableOp2?
Oauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOpOauto_encoder_4/sequential_9/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2?
Fauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOpFauto_encoder_4/sequential_9/conv2d_transpose_17/BiasAdd/ReadVariableOp2?
Oauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOpOauto_encoder_4/sequential_9/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2?
Fauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOpFauto_encoder_4/sequential_9/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
Oauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOpOauto_encoder_4/sequential_9/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2?
Fauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOpFauto_encoder_4/sequential_9/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
Oauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOpOauto_encoder_4/sequential_9/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2z
;auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp;auto_encoder_4/sequential_9/dense_14/BiasAdd/ReadVariableOp2x
:auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp:auto_encoder_4/sequential_9/dense_14/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313729

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_25_layer_call_fn_313930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3089412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313886

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_313793

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_309489

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_26_layer_call_fn_314098

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3094582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314067

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_308811

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_313469

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3103252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_309534

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_35_layer_call_and_return_conditional_losses_308758

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_309606

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_308776

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_309754

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_23_layer_call_fn_313616

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3084312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_activation_42_layer_call_fn_314269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_42_layer_call_and_return_conditional_losses_3100602
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?A
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309195

inputs
conv2d_11_309137
conv2d_11_309139!
batch_normalization_23_309142!
batch_normalization_23_309144!
batch_normalization_23_309146!
batch_normalization_23_309148
conv2d_12_309152
conv2d_12_309154!
batch_normalization_24_309157!
batch_normalization_24_309159!
batch_normalization_24_309161!
batch_normalization_24_309163
conv2d_13_309167
conv2d_13_309169!
batch_normalization_25_309172!
batch_normalization_25_309174!
batch_normalization_25_309176!
batch_normalization_25_309178
dense_12_309183
dense_12_309185
dense_13_309189
dense_13_309191
identity??.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_309137conv2d_11_309139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_3086642#
!conv2d_11/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_23_309142batch_normalization_23_309144batch_normalization_23_309146batch_normalization_23_309148*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_30869920
.batch_normalization_23/StatefulPartitionedCall?
activation_35/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_35_layer_call_and_return_conditional_losses_3087582
activation_35/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall&activation_35/PartitionedCall:output:0conv2d_12_309152conv2d_12_309154*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_3087762#
!conv2d_12/StatefulPartitionedCall?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_24_309157batch_normalization_24_309159batch_normalization_24_309161batch_normalization_24_309163*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_30881120
.batch_normalization_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_36_layer_call_and_return_conditional_losses_3088702
activation_36/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_13_309167conv2d_13_309169*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_3088882#
!conv2d_13/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_25_309172batch_normalization_25_309174batch_normalization_25_309176batch_normalization_25_309178*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_30892320
.batch_normalization_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_37_layer_call_and_return_conditional_losses_3089822
activation_37/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3089962
flatten_4/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_309183dense_12_309185*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3090142"
 dense_12/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_38_layer_call_and_return_conditional_losses_3090352
activation_38/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0dense_13_309189dense_13_309191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3090532"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_309458

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_36_layer_call_and_return_conditional_losses_313778

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_312640

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3091952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_313079
dense_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3103252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_14_input
?
?
7__inference_batch_normalization_27_layer_call_fn_314172

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3096062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
F
*__inference_reshape_4_layer_call_fn_314047

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_3099012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314215

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_activation_43_layer_call_fn_314279

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_43_layer_call_and_return_conditional_losses_3100782
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_308504

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
D__inference_dense_13_layer_call_and_return_conditional_losses_313990

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_313228

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_18_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_19_biasadd_readvariableop_resource
identity??%batch_normalization_26/AssignNewValue?'batch_normalization_26/AssignNewValue_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/BiasAdd~
activation_39/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
activation_39/Relur
reshape_4/ShapeShape activation_39/Relu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape activation_39/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape?
conv2d_transpose_16/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_16/BiasAdd?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_16/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_26/FusedBatchNormV3?
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue?
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1?
activation_40/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_40/Relu?
conv2d_transpose_17/ShapeShape activation_40/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0 activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_17/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_17/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
activation_41/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_41/Relu?
conv2d_transpose_18/ShapeShape activation_41/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice|
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/1|
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0 activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_transpose_18/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_18/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_28/FusedBatchNormV3?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1?
activation_42/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
activation_42/Relu?
conv2d_transpose_19/ShapeShape activation_42/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice|
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/1|
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0 activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_19/BiasAdd?
activation_43/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
activation_43/Sigmoid?
IdentityIdentityactivation_43/Sigmoid:y:0&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_25_layer_call_fn_313866

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3086392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_313961

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314159

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_308664

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_11_layer_call_fn_313488

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_3086642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?	
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_310854
x
sequential_8_310763
sequential_8_310765
sequential_8_310767
sequential_8_310769
sequential_8_310771
sequential_8_310773
sequential_8_310775
sequential_8_310777
sequential_8_310779
sequential_8_310781
sequential_8_310783
sequential_8_310785
sequential_8_310787
sequential_8_310789
sequential_8_310791
sequential_8_310793
sequential_8_310795
sequential_8_310797
sequential_8_310799
sequential_8_310801
sequential_8_310803
sequential_8_310805
sequential_9_310808
sequential_9_310810
sequential_9_310812
sequential_9_310814
sequential_9_310816
sequential_9_310818
sequential_9_310820
sequential_9_310822
sequential_9_310824
sequential_9_310826
sequential_9_310828
sequential_9_310830
sequential_9_310832
sequential_9_310834
sequential_9_310836
sequential_9_310838
sequential_9_310840
sequential_9_310842
sequential_9_310844
sequential_9_310846
sequential_9_310848
sequential_9_310850
identity??$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallxsequential_8_310763sequential_8_310765sequential_8_310767sequential_8_310769sequential_8_310771sequential_8_310773sequential_8_310775sequential_8_310777sequential_8_310779sequential_8_310781sequential_8_310783sequential_8_310785sequential_8_310787sequential_8_310789sequential_8_310791sequential_8_310793sequential_8_310795sequential_8_310797sequential_8_310799sequential_8_310801sequential_8_310803sequential_8_310805*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3093052&
$sequential_8/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_310808sequential_9_310810sequential_9_310812sequential_9_310814sequential_9_310816sequential_9_310818sequential_9_310820sequential_9_310822sequential_9_310824sequential_9_310826sequential_9_310828sequential_9_310830sequential_9_310832sequential_9_310834sequential_9_310836sequential_9_310838sequential_9_310840sequential_9_310842sequential_9_310844sequential_9_310846sequential_9_310848sequential_9_310850*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3103252&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
e
I__inference_activation_42_layer_call_and_return_conditional_losses_310060

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_activation_35_layer_call_fn_313626

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_35_layer_call_and_return_conditional_losses_3087582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_24_layer_call_fn_313773

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3085352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?m
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_312591

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/BiasAdd?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_23/ReadVariableOp?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_23/ReadVariableOp_1?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_23/FusedBatchNormV3?
activation_35/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation_35/Relu?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D activation_35/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/BiasAdd?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_24/ReadVariableOp?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_24/ReadVariableOp_1?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3?
activation_36/ReluRelu+batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_36/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D activation_36/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_13/BiasAdd?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3?
activation_37/ReluRelu+batch_normalization_25/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_37/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshape activation_37/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd~
activation_38/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_38/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul activation_38/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?
IdentityIdentitydense_13/BiasAdd:output:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::2p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_23_layer_call_fn_313539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3086992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_39_layer_call_and_return_conditional_losses_309879

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:?????????? 2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?J
?	
H__inference_sequential_9_layer_call_and_return_conditional_losses_310325

inputs
dense_14_310266
dense_14_310268
conv2d_transpose_16_310273
conv2d_transpose_16_310275!
batch_normalization_26_310278!
batch_normalization_26_310280!
batch_normalization_26_310282!
batch_normalization_26_310284
conv2d_transpose_17_310288
conv2d_transpose_17_310290!
batch_normalization_27_310293!
batch_normalization_27_310295!
batch_normalization_27_310297!
batch_normalization_27_310299
conv2d_transpose_18_310303
conv2d_transpose_18_310305!
batch_normalization_28_310308!
batch_normalization_28_310310!
batch_normalization_28_310312!
batch_normalization_28_310314
conv2d_transpose_19_310318
conv2d_transpose_19_310320
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_16/StatefulPartitionedCall?+conv2d_transpose_17/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_310266dense_14_310268*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_3098582"
 dense_14/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_39_layer_call_and_return_conditional_losses_3098792
activation_39/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_3099012
reshape_4/PartitionedCall?
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_16_310273conv2d_transpose_16_310275*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_3093862-
+conv2d_transpose_16/StatefulPartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0batch_normalization_26_310278batch_normalization_26_310280batch_normalization_26_310282batch_normalization_26_310284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_30948920
.batch_normalization_26/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_40_layer_call_and_return_conditional_losses_3099542
activation_40/PartitionedCall?
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall&activation_40/PartitionedCall:output:0conv2d_transpose_17_310288conv2d_transpose_17_310290*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_3095342-
+conv2d_transpose_17/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0batch_normalization_27_310293batch_normalization_27_310295batch_normalization_27_310297batch_normalization_27_310299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_30963720
.batch_normalization_27/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_41_layer_call_and_return_conditional_losses_3100072
activation_41/PartitionedCall?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall&activation_41/PartitionedCall:output:0conv2d_transpose_18_310303conv2d_transpose_18_310305*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_3096822-
+conv2d_transpose_18/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0batch_normalization_28_310308batch_normalization_28_310310batch_normalization_28_310312batch_normalization_28_310314*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_30978520
.batch_normalization_28/StatefulPartitionedCall?
activation_42/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_42_layer_call_and_return_conditional_losses_3100602
activation_42/PartitionedCall?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_transpose_19_310318conv2d_transpose_19_310320*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_3098342-
+conv2d_transpose_19/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_43_layer_call_and_return_conditional_losses_3100782
activation_43/PartitionedCall?
IdentityIdentity&activation_43/PartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_24_layer_call_fn_313696

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3088112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
/__inference_auto_encoder_4_layer_call_fn_311688
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_3108542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
J
.__inference_activation_36_layer_call_fn_313783

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_36_layer_call_and_return_conditional_losses_3088702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_17_layer_call_fn_309544

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_3095342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_28_layer_call_fn_314246

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_3097542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_25_layer_call_fn_313853

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3086082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_23_layer_call_fn_313603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3084002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_14_layer_call_fn_314018

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_3098582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_312981
dense_14_input+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_18_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_19_biasadd_readvariableop_resource
identity??6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_14_input&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/BiasAdd~
activation_39/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
activation_39/Relur
reshape_4/ShapeShape activation_39/Relu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape activation_39/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape?
conv2d_transpose_16/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_16/BiasAdd?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_16/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3?
activation_40/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_40/Relu?
conv2d_transpose_17/ShapeShape activation_40/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0 activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_17/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_17/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
activation_41/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_41/Relu?
conv2d_transpose_18/ShapeShape activation_41/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice|
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/1|
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0 activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_transpose_18/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_18/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3?
activation_42/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
activation_42/Relu?
conv2d_transpose_19/ShapeShape activation_42/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice|
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/1|
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0 activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_19/BiasAdd?
activation_43/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
activation_43/Sigmoid?	
IdentityIdentityactivation_43/Sigmoid:y:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_14_input
?
?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313904

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313665

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_activation_39_layer_call_fn_314028

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_39_layer_call_and_return_conditional_losses_3098792
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_309352
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3093052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????  ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_5
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_312838
dense_14_input+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_18_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_19_biasadd_readvariableop_resource
identity??%batch_normalization_26/AssignNewValue?'batch_normalization_26/AssignNewValue_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_16/BiasAdd/ReadVariableOp?3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?*conv2d_transpose_17/BiasAdd/ReadVariableOp?3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_14_input&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_14/BiasAdd~
activation_39/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
activation_39/Relur
reshape_4/ShapeShape activation_39/Relu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape activation_39/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape?
conv2d_transpose_16/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape?
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack?
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1?
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2?
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_16/stack/3?
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack?
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stack?
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1?
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2?
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1?
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transpose?
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOp?
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_16/BiasAdd?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_16/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_26/FusedBatchNormV3?
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue?
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1?
activation_40/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_40/Relu?
conv2d_transpose_17/ShapeShape activation_40/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape?
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack?
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1?
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2?
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_17/stack/2|
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_17/stack/3?
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack?
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stack?
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1?
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2?
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1?
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0 activation_40/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transpose?
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOp?
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_17/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_17/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
activation_41/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_41/Relu?
conv2d_transpose_18/ShapeShape activation_41/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_18/Shape?
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_18/strided_slice/stack?
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_1?
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_18/strided_slice/stack_2?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_18/strided_slice|
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/1|
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/2|
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_18/stack/3?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_18/stack?
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_18/strided_slice_1/stack?
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_1?
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_18/strided_slice_1/stack_2?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_18/strided_slice_1?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0 activation_41/Relu:activations:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2&
$conv2d_transpose_18/conv2d_transpose?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_18/BiasAdd/ReadVariableOp?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_transpose_18/BiasAdd?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_18/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_28/FusedBatchNormV3?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1?
activation_42/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
activation_42/Relu?
conv2d_transpose_19/ShapeShape activation_42/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_19/Shape?
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_19/strided_slice/stack?
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_1?
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_19/strided_slice/stack_2?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_19/strided_slice|
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/1|
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_19/stack/2|
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_19/stack/3?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_19/stack?
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_19/strided_slice_1/stack?
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_1?
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_19/strided_slice_1/stack_2?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_19/strided_slice_1?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0 activation_42/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2&
$conv2d_transpose_19/conv2d_transpose?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_19/BiasAdd/ReadVariableOp?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_19/BiasAdd?
activation_43/SigmoidSigmoid$conv2d_transpose_19/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
activation_43/Sigmoid?
IdentityIdentityactivation_43/Sigmoid:y:0&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_14_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????  D
output_18
StatefulPartitionedCall:0?????????  tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?
_tf_keras_model?{"class_name": "AutoEncoder", "name": "auto_encoder_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "AutoEncoder"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?]
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?Y
_tf_keras_sequential?Y{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?f
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
 layer_with_weights-3
 layer-6
!layer_with_weights-4
!layer-7
"layer-8
#layer_with_weights-5
#layer-9
$layer_with_weights-6
$layer-10
%layer-11
&layer_with_weights-7
&layer-12
'layer-13
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?b
_tf_keras_sequential?b{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_14_input"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_19", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_14_input"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_19", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}}}
?
,iter

-beta_1

.beta_2
	/decay
0learning_rate1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?"
	optimizer
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15
A16
B17
C18
D19
E20
F21
G22
H23
I24
J25
K26
L27
M28
N29
O30
P31"
trackable_list_wrapper
?
10
21
32
43
Q4
R5
56
67
78
89
S10
T11
912
:13
;14
<15
U16
V17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
W28
X29
G30
H31
I32
J33
Y34
Z35
K36
L37
M38
N39
[40
\41
O42
P43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

]layers
^non_trainable_variables
	variables
regularization_losses
_metrics
`layer_metrics
alayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

1kernel
2bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
?	
faxis
	3gamma
4beta
Qmoving_mean
Rmoving_variance
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}}
?	

5kernel
6bias
otrainable_variables
p	variables
qregularization_losses
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?	
saxis
	7gamma
8beta
Smoving_mean
Tmoving_variance
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
?
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}}
?	

9kernel
:bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
?	
	?axis
	;gamma
<beta
Umoving_mean
Vmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
@bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15"
trackable_list_wrapper
?
10
21
32
43
Q4
R5
56
67
78
89
S10
T11
912
:13
;14
<15
U16
V17
=18
>19
?20
@21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?layers
?non_trainable_variables
	variables
regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 25]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}}
?


Ckernel
Dbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 4, 4, 256]}}
?	
	?axis
	Egamma
Fbeta
Wmoving_mean
Xmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 8, 8, 128]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


Gkernel
Hbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 8, 8, 128]}}
?	
	?axis
	Igamma
Jbeta
Ymoving_mean
Zmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 16, 16, 64]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


Kkernel
Lbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 16, 16, 64]}}
?	
	?axis
	Mgamma
Nbeta
[moving_mean
\moving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 32]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_42", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_19", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 32]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_43", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15"
trackable_list_wrapper
?
A0
B1
C2
D3
E4
F5
W6
X7
G8
H9
I10
J11
Y12
Z13
K14
L15
M16
N17
[18
\19
O20
P21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables
?layers
?non_trainable_variables
)	variables
*regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:( 2conv2d_11/kernel
: 2conv2d_11/bias
*:( 2batch_normalization_23/gamma
):' 2batch_normalization_23/beta
*:( @2conv2d_12/kernel
:@2conv2d_12/bias
*:(@2batch_normalization_24/gamma
):'@2batch_normalization_24/beta
+:)@?2conv2d_13/kernel
:?2conv2d_13/bias
+:)?2batch_normalization_25/gamma
*:(?2batch_normalization_25/beta
#:!
??2dense_12/kernel
:?2dense_12/bias
": 	?2dense_13/kernel
:2dense_13/bias
": 	? 2dense_14/kernel
:? 2dense_14/bias
6:4??2conv2d_transpose_16/kernel
':%?2conv2d_transpose_16/bias
+:)?2batch_normalization_26/gamma
*:(?2batch_normalization_26/beta
5:3@?2conv2d_transpose_17/kernel
&:$@2conv2d_transpose_17/bias
*:(@2batch_normalization_27/gamma
):'@2batch_normalization_27/beta
4:2 @2conv2d_transpose_18/kernel
&:$ 2conv2d_transpose_18/bias
*:( 2batch_normalization_28/gamma
):' 2batch_normalization_28/beta
4:2 2conv2d_transpose_19/kernel
&:$2conv2d_transpose_19/bias
2:0  (2"batch_normalization_23/moving_mean
6:4  (2&batch_normalization_23/moving_variance
2:0@ (2"batch_normalization_24/moving_mean
6:4@ (2&batch_normalization_24/moving_variance
3:1? (2"batch_normalization_25/moving_mean
7:5? (2&batch_normalization_25/moving_variance
3:1? (2"batch_normalization_26/moving_mean
7:5? (2&batch_normalization_26/moving_variance
2:0@ (2"batch_normalization_27/moving_mean
6:4@ (2&batch_normalization_27/moving_variance
2:0  (2"batch_normalization_28/moving_mean
6:4  (2&batch_normalization_28/moving_variance
.
0
1"
trackable_list_wrapper
v
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
btrainable_variables
?layers
?non_trainable_variables
c	variables
dregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
<
30
41
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gtrainable_variables
?layers
?non_trainable_variables
h	variables
iregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ktrainable_variables
?layers
?non_trainable_variables
l	variables
mregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
otrainable_variables
?layers
?non_trainable_variables
p	variables
qregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
<
70
81
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ttrainable_variables
?layers
?non_trainable_variables
u	variables
vregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
xtrainable_variables
?layers
?non_trainable_variables
y	variables
zregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|trainable_variables
?layers
?non_trainable_variables
}	variables
~regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
<
;0
<1
U2
V3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
J
Q0
R1
S2
T3
U4
V5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
<
E0
F1
W2
X3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
<
I0
J1
Y2
Z3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
<
M0
N1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layers
?non_trainable_variables
?	variables
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13"
trackable_list_wrapper
J
W0
X1
Y2
Z3
[4
\5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_11/kernel/m
!: 2Adam/conv2d_11/bias/m
/:- 2#Adam/batch_normalization_23/gamma/m
.:, 2"Adam/batch_normalization_23/beta/m
/:- @2Adam/conv2d_12/kernel/m
!:@2Adam/conv2d_12/bias/m
/:-@2#Adam/batch_normalization_24/gamma/m
.:,@2"Adam/batch_normalization_24/beta/m
0:.@?2Adam/conv2d_13/kernel/m
": ?2Adam/conv2d_13/bias/m
0:.?2#Adam/batch_normalization_25/gamma/m
/:-?2"Adam/batch_normalization_25/beta/m
(:&
??2Adam/dense_12/kernel/m
!:?2Adam/dense_12/bias/m
':%	?2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
':%	? 2Adam/dense_14/kernel/m
!:? 2Adam/dense_14/bias/m
;:9??2!Adam/conv2d_transpose_16/kernel/m
,:*?2Adam/conv2d_transpose_16/bias/m
0:.?2#Adam/batch_normalization_26/gamma/m
/:-?2"Adam/batch_normalization_26/beta/m
::8@?2!Adam/conv2d_transpose_17/kernel/m
+:)@2Adam/conv2d_transpose_17/bias/m
/:-@2#Adam/batch_normalization_27/gamma/m
.:,@2"Adam/batch_normalization_27/beta/m
9:7 @2!Adam/conv2d_transpose_18/kernel/m
+:) 2Adam/conv2d_transpose_18/bias/m
/:- 2#Adam/batch_normalization_28/gamma/m
.:, 2"Adam/batch_normalization_28/beta/m
9:7 2!Adam/conv2d_transpose_19/kernel/m
+:)2Adam/conv2d_transpose_19/bias/m
/:- 2Adam/conv2d_11/kernel/v
!: 2Adam/conv2d_11/bias/v
/:- 2#Adam/batch_normalization_23/gamma/v
.:, 2"Adam/batch_normalization_23/beta/v
/:- @2Adam/conv2d_12/kernel/v
!:@2Adam/conv2d_12/bias/v
/:-@2#Adam/batch_normalization_24/gamma/v
.:,@2"Adam/batch_normalization_24/beta/v
0:.@?2Adam/conv2d_13/kernel/v
": ?2Adam/conv2d_13/bias/v
0:.?2#Adam/batch_normalization_25/gamma/v
/:-?2"Adam/batch_normalization_25/beta/v
(:&
??2Adam/dense_12/kernel/v
!:?2Adam/dense_12/bias/v
':%	?2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
':%	? 2Adam/dense_14/kernel/v
!:? 2Adam/dense_14/bias/v
;:9??2!Adam/conv2d_transpose_16/kernel/v
,:*?2Adam/conv2d_transpose_16/bias/v
0:.?2#Adam/batch_normalization_26/gamma/v
/:-?2"Adam/batch_normalization_26/beta/v
::8@?2!Adam/conv2d_transpose_17/kernel/v
+:)@2Adam/conv2d_transpose_17/bias/v
/:-@2#Adam/batch_normalization_27/gamma/v
.:,@2"Adam/batch_normalization_27/beta/v
9:7 @2!Adam/conv2d_transpose_18/kernel/v
+:) 2Adam/conv2d_transpose_18/bias/v
/:- 2#Adam/batch_normalization_28/gamma/v
.:, 2"Adam/batch_normalization_28/beta/v
9:7 2!Adam/conv2d_transpose_19/kernel/v
+:)2Adam/conv2d_transpose_19/bias/v
?2?
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311595
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311374
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312235
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312014?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
!__inference__wrapped_model_308338?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????  
?2?
/__inference_auto_encoder_4_layer_call_fn_312328
/__inference_auto_encoder_4_layer_call_fn_311688
/__inference_auto_encoder_4_layer_call_fn_311781
/__inference_auto_encoder_4_layer_call_fn_312421?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309131
H__inference_sequential_8_layer_call_and_return_conditional_losses_312591
H__inference_sequential_8_layer_call_and_return_conditional_losses_312509
H__inference_sequential_8_layer_call_and_return_conditional_losses_309070?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_8_layer_call_fn_312640
-__inference_sequential_8_layer_call_fn_312689
-__inference_sequential_8_layer_call_fn_309352
-__inference_sequential_8_layer_call_fn_309242?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_9_layer_call_and_return_conditional_losses_312838
H__inference_sequential_9_layer_call_and_return_conditional_losses_313371
H__inference_sequential_9_layer_call_and_return_conditional_losses_313228
H__inference_sequential_9_layer_call_and_return_conditional_losses_312981?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_9_layer_call_fn_313030
-__inference_sequential_9_layer_call_fn_313079
-__inference_sequential_9_layer_call_fn_313420
-__inference_sequential_9_layer_call_fn_313469?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_311141input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_313479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_11_layer_call_fn_313488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313508
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313526
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313572
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313590?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_23_layer_call_fn_313616
7__inference_batch_normalization_23_layer_call_fn_313603
7__inference_batch_normalization_23_layer_call_fn_313539
7__inference_batch_normalization_23_layer_call_fn_313552?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_35_layer_call_and_return_conditional_losses_313621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_35_layer_call_fn_313626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_313636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_12_layer_call_fn_313645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313747
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313729
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313665
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313683?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_24_layer_call_fn_313760
7__inference_batch_normalization_24_layer_call_fn_313773
7__inference_batch_normalization_24_layer_call_fn_313709
7__inference_batch_normalization_24_layer_call_fn_313696?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_36_layer_call_and_return_conditional_losses_313778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_36_layer_call_fn_313783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_313793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_13_layer_call_fn_313802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313840
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313822
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313886
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313904?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_25_layer_call_fn_313866
7__inference_batch_normalization_25_layer_call_fn_313853
7__inference_batch_normalization_25_layer_call_fn_313930
7__inference_batch_normalization_25_layer_call_fn_313917?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_37_layer_call_and_return_conditional_losses_313935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_37_layer_call_fn_313940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_4_layer_call_and_return_conditional_losses_313946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_4_layer_call_fn_313951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_313961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_313970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_38_layer_call_and_return_conditional_losses_313975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_38_layer_call_fn_313980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_13_layer_call_and_return_conditional_losses_313990?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_13_layer_call_fn_313999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_14_layer_call_and_return_conditional_losses_314009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_14_layer_call_fn_314018?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_39_layer_call_and_return_conditional_losses_314023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_39_layer_call_fn_314028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_4_layer_call_and_return_conditional_losses_314042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_4_layer_call_fn_314047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_309386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
4__inference_conv2d_transpose_16_layer_call_fn_309396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314067
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314085?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_26_layer_call_fn_314098
7__inference_batch_normalization_26_layer_call_fn_314111?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_40_layer_call_and_return_conditional_losses_314116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_40_layer_call_fn_314121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_309534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
4__inference_conv2d_transpose_17_layer_call_fn_309544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314159
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314141?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_27_layer_call_fn_314185
7__inference_batch_normalization_27_layer_call_fn_314172?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_41_layer_call_and_return_conditional_losses_314190?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_41_layer_call_fn_314195?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_309682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
4__inference_conv2d_transpose_18_layer_call_fn_309692?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314233
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314215?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_28_layer_call_fn_314259
7__inference_batch_normalization_28_layer_call_fn_314246?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_42_layer_call_and_return_conditional_losses_314264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_42_layer_call_fn_314269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_309834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
4__inference_conv2d_transpose_19_layer_call_fn_309844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
I__inference_activation_43_layer_call_and_return_conditional_losses_314274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_43_layer_call_fn_314279?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_308338?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP8?5
.?+
)?&
input_1?????????  
? ";?8
6
output_1*?'
output_1?????????  ?
I__inference_activation_35_layer_call_and_return_conditional_losses_313621h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_activation_35_layer_call_fn_313626[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_activation_36_layer_call_and_return_conditional_losses_313778h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_activation_36_layer_call_fn_313783[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
I__inference_activation_37_layer_call_and_return_conditional_losses_313935j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_37_layer_call_fn_313940]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_38_layer_call_and_return_conditional_losses_313975Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
.__inference_activation_38_layer_call_fn_313980M0?-
&?#
!?
inputs??????????
? "????????????
I__inference_activation_39_layer_call_and_return_conditional_losses_314023Z0?-
&?#
!?
inputs?????????? 
? "&?#
?
0?????????? 
? 
.__inference_activation_39_layer_call_fn_314028M0?-
&?#
!?
inputs?????????? 
? "??????????? ?
I__inference_activation_40_layer_call_and_return_conditional_losses_314116?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_activation_40_layer_call_fn_314121?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_activation_41_layer_call_and_return_conditional_losses_314190?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_activation_41_layer_call_fn_314195I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
I__inference_activation_42_layer_call_and_return_conditional_losses_314264?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
.__inference_activation_42_layer_call_fn_314269I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
I__inference_activation_43_layer_call_and_return_conditional_losses_314274?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
.__inference_activation_43_layer_call_fn_314279I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311374?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP<?9
2?/
)?&
input_1?????????  
p
? "-?*
#? 
0?????????  
? ?
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_311595?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP<?9
2?/
)?&
input_1?????????  
p 
? "-?*
#? 
0?????????  
? ?
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312014?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP6?3
,?)
#? 
x?????????  
p
? "-?*
#? 
0?????????  
? ?
J__inference_auto_encoder_4_layer_call_and_return_conditional_losses_312235?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP6?3
,?)
#? 
x?????????  
p 
? "-?*
#? 
0?????????  
? ?
/__inference_auto_encoder_4_layer_call_fn_311688?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP<?9
2?/
)?&
input_1?????????  
p
? "2?/+????????????????????????????
/__inference_auto_encoder_4_layer_call_fn_311781?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP<?9
2?/
)?&
input_1?????????  
p 
? "2?/+????????????????????????????
/__inference_auto_encoder_4_layer_call_fn_312328?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP6?3
,?)
#? 
x?????????  
p
? "2?/+????????????????????????????
/__inference_auto_encoder_4_layer_call_fn_312421?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OP6?3
,?)
#? 
x?????????  
p 
? "2?/+????????????????????????????
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313508r34QR;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313526r34QR;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313572?34QRM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_313590?34QRM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_23_layer_call_fn_313539e34QR;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
7__inference_batch_normalization_23_layer_call_fn_313552e34QR;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
7__inference_batch_normalization_23_layer_call_fn_313603?34QRM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_23_layer_call_fn_313616?34QRM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313665r78ST;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313683r78ST;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313729?78STM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_313747?78STM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_24_layer_call_fn_313696e78ST;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
7__inference_batch_normalization_24_layer_call_fn_313709e78ST;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
7__inference_batch_normalization_24_layer_call_fn_313760?78STM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_24_layer_call_fn_313773?78STM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313822?;<UVN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313840?;<UVN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313886t;<UV<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_313904t;<UV<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_25_layer_call_fn_313853?;<UVN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_25_layer_call_fn_313866?;<UVN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_25_layer_call_fn_313917g;<UV<?9
2?/
)?&
inputs??????????
p
? "!????????????
7__inference_batch_normalization_25_layer_call_fn_313930g;<UV<?9
2?/
)?&
inputs??????????
p 
? "!????????????
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314067?EFWXN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_314085?EFWXN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
7__inference_batch_normalization_26_layer_call_fn_314098?EFWXN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_26_layer_call_fn_314111?EFWXN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314141?IJYZM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_314159?IJYZM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_27_layer_call_fn_314172?IJYZM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_27_layer_call_fn_314185?IJYZM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314215?MN[\M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_314233?MN[\M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_28_layer_call_fn_314246?MN[\M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_28_layer_call_fn_314259?MN[\M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_313479l127?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_11_layer_call_fn_313488_127?4
-?*
(?%
inputs?????????  
? " ?????????? ?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_313636l567?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_12_layer_call_fn_313645_567?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_313793m9:7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_13_layer_call_fn_313802`9:7?4
-?*
(?%
inputs?????????@
? "!????????????
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_309386?CDJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_conv2d_transpose_16_layer_call_fn_309396?CDJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_309534?GHJ?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
4__inference_conv2d_transpose_17_layer_call_fn_309544?GHJ?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_309682?KLI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_conv2d_transpose_18_layer_call_fn_309692?KLI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
O__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_309834?OPI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_19_layer_call_fn_309844?OPI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_dense_12_layer_call_and_return_conditional_losses_313961^=>0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_12_layer_call_fn_313970Q=>0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_13_layer_call_and_return_conditional_losses_313990]?@0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_13_layer_call_fn_313999P?@0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_14_layer_call_and_return_conditional_losses_314009]AB/?,
%?"
 ?
inputs?????????
? "&?#
?
0?????????? 
? }
)__inference_dense_14_layer_call_fn_314018PAB/?,
%?"
 ?
inputs?????????
? "??????????? ?
E__inference_flatten_4_layer_call_and_return_conditional_losses_313946b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
*__inference_flatten_4_layer_call_fn_313951U8?5
.?+
)?&
inputs??????????
? "????????????
E__inference_reshape_4_layer_call_and_return_conditional_losses_314042b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
*__inference_reshape_4_layer_call_fn_314047U0?-
&?#
!?
inputs?????????? 
? "!????????????
H__inference_sequential_8_layer_call_and_return_conditional_losses_309070?1234QR5678ST9:;<UV=>?@@?=
6?3
)?&
input_5?????????  
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_309131?1234QR5678ST9:;<UV=>?@@?=
6?3
)?&
input_5?????????  
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_312509?1234QR5678ST9:;<UV=>?@??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_312591?1234QR5678ST9:;<UV=>?@??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_8_layer_call_fn_309242t1234QR5678ST9:;<UV=>?@@?=
6?3
)?&
input_5?????????  
p

 
? "???????????
-__inference_sequential_8_layer_call_fn_309352t1234QR5678ST9:;<UV=>?@@?=
6?3
)?&
input_5?????????  
p 

 
? "???????????
-__inference_sequential_8_layer_call_fn_312640s1234QR5678ST9:;<UV=>?@??<
5?2
(?%
inputs?????????  
p

 
? "???????????
-__inference_sequential_8_layer_call_fn_312689s1234QR5678ST9:;<UV=>?@??<
5?2
(?%
inputs?????????  
p 

 
? "???????????
H__inference_sequential_9_layer_call_and_return_conditional_losses_312838?ABCDEFWXGHIJYZKLMN[\OP??<
5?2
(?%
dense_14_input?????????
p

 
? "-?*
#? 
0?????????  
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_312981?ABCDEFWXGHIJYZKLMN[\OP??<
5?2
(?%
dense_14_input?????????
p 

 
? "-?*
#? 
0?????????  
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_313228?ABCDEFWXGHIJYZKLMN[\OP7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????  
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_313371?ABCDEFWXGHIJYZKLMN[\OP7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????  
? ?
-__inference_sequential_9_layer_call_fn_313030?ABCDEFWXGHIJYZKLMN[\OP??<
5?2
(?%
dense_14_input?????????
p

 
? "2?/+????????????????????????????
-__inference_sequential_9_layer_call_fn_313079?ABCDEFWXGHIJYZKLMN[\OP??<
5?2
(?%
dense_14_input?????????
p 

 
? "2?/+????????????????????????????
-__inference_sequential_9_layer_call_fn_313420?ABCDEFWXGHIJYZKLMN[\OP7?4
-?*
 ?
inputs?????????
p

 
? "2?/+????????????????????????????
-__inference_sequential_9_layer_call_fn_313469?ABCDEFWXGHIJYZKLMN[\OP7?4
-?*
 ?
inputs?????????
p 

 
? "2?/+????????????????????????????
$__inference_signature_wrapper_311141?,1234QR5678ST9:;<UV=>?@ABCDEFWXGHIJYZKLMN[\OPC?@
? 
9?6
4
input_1)?&
input_1?????????  ";?8
6
output_1*?'
output_1?????????  