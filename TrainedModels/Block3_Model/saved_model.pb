??"
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-0-ge5bf8de4108??
?
block3__net/Block_D1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameblock3__net/Block_D1/kernel
?
/block3__net/Block_D1/kernel/Read/ReadVariableOpReadVariableOpblock3__net/Block_D1/kernel*&
_output_shapes
:@*
dtype0
?
block3__net/Block_D1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameblock3__net/Block_D1/bias
?
-block3__net/Block_D1/bias/Read/ReadVariableOpReadVariableOpblock3__net/Block_D1/bias*
_output_shapes
:*
dtype0
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
%block3__net/sequential/block30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*6
shared_name'%block3__net/sequential/block30/kernel
?
9block3__net/sequential/block30/kernel/Read/ReadVariableOpReadVariableOp%block3__net/sequential/block30/kernel*(
_output_shapes
:??*
dtype0
?
#block3__net/sequential/block30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#block3__net/sequential/block30/bias
?
7block3__net/sequential/block30/bias/Read/ReadVariableOpReadVariableOp#block3__net/sequential/block30/bias*
_output_shapes	
:?*
dtype0
?
.block3__net/sequential/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*?
shared_name0.block3__net/sequential/conv2d_transpose/kernel
?
Bblock3__net/sequential/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp.block3__net/sequential/conv2d_transpose/kernel*(
_output_shapes
:??*
dtype0
?
,block3__net/sequential/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,block3__net/sequential/conv2d_transpose/bias
?
@block3__net/sequential/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp,block3__net/sequential/conv2d_transpose/bias*
_output_shapes	
:?*
dtype0
?
%block3__net/sequential/block20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*6
shared_name'%block3__net/sequential/block20/kernel
?
9block3__net/sequential/block20/kernel/Read/ReadVariableOpReadVariableOp%block3__net/sequential/block20/kernel*(
_output_shapes
:??*
dtype0
?
#block3__net/sequential/block20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#block3__net/sequential/block20/bias
?
7block3__net/sequential/block20/bias/Read/ReadVariableOpReadVariableOp#block3__net/sequential/block20/bias*
_output_shapes	
:?*
dtype0
?
%block3__net/sequential/block21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*6
shared_name'%block3__net/sequential/block21/kernel
?
9block3__net/sequential/block21/kernel/Read/ReadVariableOpReadVariableOp%block3__net/sequential/block21/kernel*(
_output_shapes
:??*
dtype0
?
#block3__net/sequential/block21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#block3__net/sequential/block21/bias
?
7block3__net/sequential/block21/bias/Read/ReadVariableOpReadVariableOp#block3__net/sequential/block21/bias*
_output_shapes	
:?*
dtype0
?
%block3__net/sequential/block22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*6
shared_name'%block3__net/sequential/block22/kernel
?
9block3__net/sequential/block22/kernel/Read/ReadVariableOpReadVariableOp%block3__net/sequential/block22/kernel*(
_output_shapes
:??*
dtype0
?
#block3__net/sequential/block22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#block3__net/sequential/block22/bias
?
7block3__net/sequential/block22/bias/Read/ReadVariableOpReadVariableOp#block3__net/sequential/block22/bias*
_output_shapes	
:?*
dtype0
?
0block3__net/sequential/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*A
shared_name20block3__net/sequential/conv2d_transpose_1/kernel
?
Dblock3__net/sequential/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp0block3__net/sequential/conv2d_transpose_1/kernel*'
_output_shapes
:@?*
dtype0
?
.block3__net/sequential/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.block3__net/sequential/conv2d_transpose_1/bias
?
Bblock3__net/sequential/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp.block3__net/sequential/conv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
?
%block3__net/sequential/block10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%block3__net/sequential/block10/kernel
?
9block3__net/sequential/block10/kernel/Read/ReadVariableOpReadVariableOp%block3__net/sequential/block10/kernel*&
_output_shapes
:@@*
dtype0
?
#block3__net/sequential/block10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#block3__net/sequential/block10/bias
?
7block3__net/sequential/block10/bias/Read/ReadVariableOpReadVariableOp#block3__net/sequential/block10/bias*
_output_shapes
:@*
dtype0
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
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
"Adam/block3__net/Block_D1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/block3__net/Block_D1/kernel/m
?
6Adam/block3__net/Block_D1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/block3__net/Block_D1/kernel/m*&
_output_shapes
:@*
dtype0
?
 Adam/block3__net/Block_D1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/block3__net/Block_D1/bias/m
?
4Adam/block3__net/Block_D1/bias/m/Read/ReadVariableOpReadVariableOp Adam/block3__net/Block_D1/bias/m*
_output_shapes
:*
dtype0
?
,Adam/block3__net/sequential/block30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block30/kernel/m
?
@Adam/block3__net/sequential/block30/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block30/kernel/m*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block30/bias/m
?
>Adam/block3__net/sequential/block30/bias/m/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block30/bias/m*
_output_shapes	
:?*
dtype0
?
5Adam/block3__net/sequential/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*F
shared_name75Adam/block3__net/sequential/conv2d_transpose/kernel/m
?
IAdam/block3__net/sequential/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/block3__net/sequential/conv2d_transpose/kernel/m*(
_output_shapes
:??*
dtype0
?
3Adam/block3__net/sequential/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53Adam/block3__net/sequential/conv2d_transpose/bias/m
?
GAdam/block3__net/sequential/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp3Adam/block3__net/sequential/conv2d_transpose/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block20/kernel/m
?
@Adam/block3__net/sequential/block20/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block20/kernel/m*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block20/bias/m
?
>Adam/block3__net/sequential/block20/bias/m/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block20/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block21/kernel/m
?
@Adam/block3__net/sequential/block21/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block21/kernel/m*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block21/bias/m
?
>Adam/block3__net/sequential/block21/bias/m/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block21/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block22/kernel/m
?
@Adam/block3__net/sequential/block22/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block22/kernel/m*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block22/bias/m
?
>Adam/block3__net/sequential/block22/bias/m/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block22/bias/m*
_output_shapes	
:?*
dtype0
?
7Adam/block3__net/sequential/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*H
shared_name97Adam/block3__net/sequential/conv2d_transpose_1/kernel/m
?
KAdam/block3__net/sequential/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp7Adam/block3__net/sequential/conv2d_transpose_1/kernel/m*'
_output_shapes
:@?*
dtype0
?
5Adam/block3__net/sequential/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/block3__net/sequential/conv2d_transpose_1/bias/m
?
IAdam/block3__net/sequential/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp5Adam/block3__net/sequential/conv2d_transpose_1/bias/m*
_output_shapes
:@*
dtype0
?
,Adam/block3__net/sequential/block10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/block3__net/sequential/block10/kernel/m
?
@Adam/block3__net/sequential/block10/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block10/kernel/m*&
_output_shapes
:@@*
dtype0
?
*Adam/block3__net/sequential/block10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/block3__net/sequential/block10/bias/m
?
>Adam/block3__net/sequential/block10/bias/m/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block10/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/block3__net/Block_D1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/block3__net/Block_D1/kernel/v
?
6Adam/block3__net/Block_D1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/block3__net/Block_D1/kernel/v*&
_output_shapes
:@*
dtype0
?
 Adam/block3__net/Block_D1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/block3__net/Block_D1/bias/v
?
4Adam/block3__net/Block_D1/bias/v/Read/ReadVariableOpReadVariableOp Adam/block3__net/Block_D1/bias/v*
_output_shapes
:*
dtype0
?
,Adam/block3__net/sequential/block30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block30/kernel/v
?
@Adam/block3__net/sequential/block30/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block30/kernel/v*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block30/bias/v
?
>Adam/block3__net/sequential/block30/bias/v/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block30/bias/v*
_output_shapes	
:?*
dtype0
?
5Adam/block3__net/sequential/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*F
shared_name75Adam/block3__net/sequential/conv2d_transpose/kernel/v
?
IAdam/block3__net/sequential/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/block3__net/sequential/conv2d_transpose/kernel/v*(
_output_shapes
:??*
dtype0
?
3Adam/block3__net/sequential/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53Adam/block3__net/sequential/conv2d_transpose/bias/v
?
GAdam/block3__net/sequential/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp3Adam/block3__net/sequential/conv2d_transpose/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block20/kernel/v
?
@Adam/block3__net/sequential/block20/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block20/kernel/v*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block20/bias/v
?
>Adam/block3__net/sequential/block20/bias/v/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block20/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block21/kernel/v
?
@Adam/block3__net/sequential/block21/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block21/kernel/v*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block21/bias/v
?
>Adam/block3__net/sequential/block21/bias/v/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block21/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/block3__net/sequential/block22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*=
shared_name.,Adam/block3__net/sequential/block22/kernel/v
?
@Adam/block3__net/sequential/block22/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block22/kernel/v*(
_output_shapes
:??*
dtype0
?
*Adam/block3__net/sequential/block22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/block3__net/sequential/block22/bias/v
?
>Adam/block3__net/sequential/block22/bias/v/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block22/bias/v*
_output_shapes	
:?*
dtype0
?
7Adam/block3__net/sequential/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*H
shared_name97Adam/block3__net/sequential/conv2d_transpose_1/kernel/v
?
KAdam/block3__net/sequential/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp7Adam/block3__net/sequential/conv2d_transpose_1/kernel/v*'
_output_shapes
:@?*
dtype0
?
5Adam/block3__net/sequential/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/block3__net/sequential/conv2d_transpose_1/bias/v
?
IAdam/block3__net/sequential/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp5Adam/block3__net/sequential/conv2d_transpose_1/bias/v*
_output_shapes
:@*
dtype0
?
,Adam/block3__net/sequential/block10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/block3__net/sequential/block10/kernel/v
?
@Adam/block3__net/sequential/block10/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/block3__net/sequential/block10/kernel/v*&
_output_shapes
:@@*
dtype0
?
*Adam/block3__net/sequential/block10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/block3__net/sequential/block10/bias/v
?
>Adam/block3__net/sequential/block10/bias/v/Read/ReadVariableOpReadVariableOp*Adam/block3__net/sequential/block10/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?v
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?v
value?uB?u B?u
?
encoder_Model
decoder_Model
conv_r1
	optimizer
_training_endpoints
regularization_losses
trainable_variables
	variables
		keras_api


signatures
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?
(iter

)beta_1

*beta_2
	+decay
,learning_rate"m?#m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?"v?#v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?
 
 
v
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
"14
#15
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
-10
.11
/12
013
114
215
316
417
518
619
720
821
922
:23
"24
#25
?
regularization_losses

Elayers
Flayer_regularization_losses
Gmetrics
trainable_variables
Hnon_trainable_variables
	variables
 
 
h

;kernel
<bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
h

=kernel
>bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
R
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

?kernel
@bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
h

Akernel
Bbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
R
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

Ckernel
Dbias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
 
 
F
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
?
regularization_losses

elayers
flayer_regularization_losses
gmetrics
trainable_variables
hnon_trainable_variables
	variables
h

-kernel
.bias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

/kernel
0bias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
h

1kernel
2bias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
h

3kernel
4bias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
h

5kernel
6bias
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
i

7kernel
8bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
l

9kernel
:bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
f
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
f
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
?
regularization_losses
?layers
 ?layer_regularization_losses
?metrics
trainable_variables
?non_trainable_variables
 	variables
ZX
VARIABLE_VALUEblock3__net/Block_D1/kernel)conv_r1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEblock3__net/Block_D1/bias'conv_r1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
$regularization_losses
?layers
 ?layer_regularization_losses
?metrics
%trainable_variables
?non_trainable_variables
&	variables
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
ki
VARIABLE_VALUE%block3__net/sequential/block30/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#block3__net/sequential/block30/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE.block3__net/sequential/conv2d_transpose/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE,block3__net/sequential/conv2d_transpose/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%block3__net/sequential/block20/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#block3__net/sequential/block20/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%block3__net/sequential/block21/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#block3__net/sequential/block21/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%block3__net/sequential/block22/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#block3__net/sequential/block22/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0block3__net/sequential/conv2d_transpose_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.block3__net/sequential/conv2d_transpose_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%block3__net/sequential/block10/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#block3__net/sequential/block10/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

?0
F
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
 
 

;0
<1
?
Iregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Jtrainable_variables
?non_trainable_variables
K	variables
 
 

=0
>1
?
Mregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Ntrainable_variables
?non_trainable_variables
O	variables
 
 
 
?
Qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Rtrainable_variables
?non_trainable_variables
S	variables
 
 

?0
@1
?
Uregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Vtrainable_variables
?non_trainable_variables
W	variables
 
 

A0
B1
?
Yregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Ztrainable_variables
?non_trainable_variables
[	variables
 
 
 
?
]regularization_losses
?layers
 ?layer_regularization_losses
?metrics
^trainable_variables
?non_trainable_variables
_	variables
 
 

C0
D1
?
aregularization_losses
?layers
 ?layer_regularization_losses
?metrics
btrainable_variables
?non_trainable_variables
c	variables
8
0
1
2
3
4
5
6
7
 
 
F
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
 

-0
.1

-0
.1
?
iregularization_losses
?layers
 ?layer_regularization_losses
?metrics
jtrainable_variables
?non_trainable_variables
k	variables
 

/0
01

/0
01
?
mregularization_losses
?layers
 ?layer_regularization_losses
?metrics
ntrainable_variables
?non_trainable_variables
o	variables
 

10
21

10
21
?
qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
rtrainable_variables
?non_trainable_variables
s	variables
 

30
41

30
41
?
uregularization_losses
?layers
 ?layer_regularization_losses
?metrics
vtrainable_variables
?non_trainable_variables
w	variables
 

50
61

50
61
?
yregularization_losses
?layers
 ?layer_regularization_losses
?metrics
ztrainable_variables
?non_trainable_variables
{	variables
 

70
81

70
81
?
}regularization_losses
?layers
 ?layer_regularization_losses
?metrics
~trainable_variables
?non_trainable_variables
	variables
 

90
:1

90
:1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?trainable_variables
?non_trainable_variables
?	variables
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 

;0
<1
 
 
 

=0
>1
 
 
 
 
 
 
 

?0
@1
 
 
 

A0
B1
 
 
 
 
 
 
 

C0
D1
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
?
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?trainable_variables
?non_trainable_variables
?	variables
 
 
 

?0
?1
}{
VARIABLE_VALUE"Adam/block3__net/Block_D1/kernel/mEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/block3__net/Block_D1/bias/mCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block30/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block30/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/block3__net/sequential/conv2d_transpose/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/block3__net/sequential/conv2d_transpose/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block20/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block20/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block21/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block21/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block22/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block22/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/block3__net/sequential/conv2d_transpose_1/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/block3__net/sequential/conv2d_transpose_1/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block10/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block10/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE"Adam/block3__net/Block_D1/kernel/vEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/block3__net/Block_D1/bias/vCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block30/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block30/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/block3__net/sequential/conv2d_transpose/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/block3__net/sequential/conv2d_transpose/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block20/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block20/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block21/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block21/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block22/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block22/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/block3__net/sequential/conv2d_transpose_1/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/block3__net/sequential/conv2d_transpose_1/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/block3__net/sequential/block10/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/block3__net/sequential/block10/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/bias%block3__net/sequential/block30/kernel#block3__net/sequential/block30/bias.block3__net/sequential/conv2d_transpose/kernel,block3__net/sequential/conv2d_transpose/bias%block3__net/sequential/block20/kernel#block3__net/sequential/block20/bias%block3__net/sequential/block21/kernel#block3__net/sequential/block21/bias%block3__net/sequential/block22/kernel#block3__net/sequential/block22/bias0block3__net/sequential/conv2d_transpose_1/kernel.block3__net/sequential/conv2d_transpose_1/bias%block3__net/sequential/block10/kernel#block3__net/sequential/block10/biasblock3__net/Block_D1/kernelblock3__net/Block_D1/bias*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:???????????*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_43941
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/block3__net/Block_D1/kernel/Read/ReadVariableOp-block3__net/Block_D1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9block3__net/sequential/block30/kernel/Read/ReadVariableOp7block3__net/sequential/block30/bias/Read/ReadVariableOpBblock3__net/sequential/conv2d_transpose/kernel/Read/ReadVariableOp@block3__net/sequential/conv2d_transpose/bias/Read/ReadVariableOp9block3__net/sequential/block20/kernel/Read/ReadVariableOp7block3__net/sequential/block20/bias/Read/ReadVariableOp9block3__net/sequential/block21/kernel/Read/ReadVariableOp7block3__net/sequential/block21/bias/Read/ReadVariableOp9block3__net/sequential/block22/kernel/Read/ReadVariableOp7block3__net/sequential/block22/bias/Read/ReadVariableOpDblock3__net/sequential/conv2d_transpose_1/kernel/Read/ReadVariableOpBblock3__net/sequential/conv2d_transpose_1/bias/Read/ReadVariableOp9block3__net/sequential/block10/kernel/Read/ReadVariableOp7block3__net/sequential/block10/bias/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/block3__net/Block_D1/kernel/m/Read/ReadVariableOp4Adam/block3__net/Block_D1/bias/m/Read/ReadVariableOp@Adam/block3__net/sequential/block30/kernel/m/Read/ReadVariableOp>Adam/block3__net/sequential/block30/bias/m/Read/ReadVariableOpIAdam/block3__net/sequential/conv2d_transpose/kernel/m/Read/ReadVariableOpGAdam/block3__net/sequential/conv2d_transpose/bias/m/Read/ReadVariableOp@Adam/block3__net/sequential/block20/kernel/m/Read/ReadVariableOp>Adam/block3__net/sequential/block20/bias/m/Read/ReadVariableOp@Adam/block3__net/sequential/block21/kernel/m/Read/ReadVariableOp>Adam/block3__net/sequential/block21/bias/m/Read/ReadVariableOp@Adam/block3__net/sequential/block22/kernel/m/Read/ReadVariableOp>Adam/block3__net/sequential/block22/bias/m/Read/ReadVariableOpKAdam/block3__net/sequential/conv2d_transpose_1/kernel/m/Read/ReadVariableOpIAdam/block3__net/sequential/conv2d_transpose_1/bias/m/Read/ReadVariableOp@Adam/block3__net/sequential/block10/kernel/m/Read/ReadVariableOp>Adam/block3__net/sequential/block10/bias/m/Read/ReadVariableOp6Adam/block3__net/Block_D1/kernel/v/Read/ReadVariableOp4Adam/block3__net/Block_D1/bias/v/Read/ReadVariableOp@Adam/block3__net/sequential/block30/kernel/v/Read/ReadVariableOp>Adam/block3__net/sequential/block30/bias/v/Read/ReadVariableOpIAdam/block3__net/sequential/conv2d_transpose/kernel/v/Read/ReadVariableOpGAdam/block3__net/sequential/conv2d_transpose/bias/v/Read/ReadVariableOp@Adam/block3__net/sequential/block20/kernel/v/Read/ReadVariableOp>Adam/block3__net/sequential/block20/bias/v/Read/ReadVariableOp@Adam/block3__net/sequential/block21/kernel/v/Read/ReadVariableOp>Adam/block3__net/sequential/block21/bias/v/Read/ReadVariableOp@Adam/block3__net/sequential/block22/kernel/v/Read/ReadVariableOp>Adam/block3__net/sequential/block22/bias/v/Read/ReadVariableOpKAdam/block3__net/sequential/conv2d_transpose_1/kernel/v/Read/ReadVariableOpIAdam/block3__net/sequential/conv2d_transpose_1/bias/v/Read/ReadVariableOp@Adam/block3__net/sequential/block10/kernel/v/Read/ReadVariableOp>Adam/block3__net/sequential/block10/bias/v/Read/ReadVariableOpConst*N
TinG
E2C	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_45096
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock3__net/Block_D1/kernelblock3__net/Block_D1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%block3__net/sequential/block30/kernel#block3__net/sequential/block30/bias.block3__net/sequential/conv2d_transpose/kernel,block3__net/sequential/conv2d_transpose/bias%block3__net/sequential/block20/kernel#block3__net/sequential/block20/bias%block3__net/sequential/block21/kernel#block3__net/sequential/block21/bias%block3__net/sequential/block22/kernel#block3__net/sequential/block22/bias0block3__net/sequential/conv2d_transpose_1/kernel.block3__net/sequential/conv2d_transpose_1/bias%block3__net/sequential/block10/kernel#block3__net/sequential/block10/biasblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biastotalcount"Adam/block3__net/Block_D1/kernel/m Adam/block3__net/Block_D1/bias/m,Adam/block3__net/sequential/block30/kernel/m*Adam/block3__net/sequential/block30/bias/m5Adam/block3__net/sequential/conv2d_transpose/kernel/m3Adam/block3__net/sequential/conv2d_transpose/bias/m,Adam/block3__net/sequential/block20/kernel/m*Adam/block3__net/sequential/block20/bias/m,Adam/block3__net/sequential/block21/kernel/m*Adam/block3__net/sequential/block21/bias/m,Adam/block3__net/sequential/block22/kernel/m*Adam/block3__net/sequential/block22/bias/m7Adam/block3__net/sequential/conv2d_transpose_1/kernel/m5Adam/block3__net/sequential/conv2d_transpose_1/bias/m,Adam/block3__net/sequential/block10/kernel/m*Adam/block3__net/sequential/block10/bias/m"Adam/block3__net/Block_D1/kernel/v Adam/block3__net/Block_D1/bias/v,Adam/block3__net/sequential/block30/kernel/v*Adam/block3__net/sequential/block30/bias/v5Adam/block3__net/sequential/conv2d_transpose/kernel/v3Adam/block3__net/sequential/conv2d_transpose/bias/v,Adam/block3__net/sequential/block20/kernel/v*Adam/block3__net/sequential/block20/bias/v,Adam/block3__net/sequential/block21/kernel/v*Adam/block3__net/sequential/block21/bias/v,Adam/block3__net/sequential/block22/kernel/v*Adam/block3__net/sequential/block22/bias/v7Adam/block3__net/sequential/conv2d_transpose_1/kernel/v5Adam/block3__net/sequential/conv2d_transpose_1/bias/v,Adam/block3__net/sequential/block10/kernel/v*Adam/block3__net/sequential/block10/bias/v*M
TinF
D2B*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_45303??
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_42917

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_1_layer_call_fn_43242

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_432342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
΍
?

E__inference_sequential_layer_call_and_return_conditional_losses_44742

inputs*
&block30_conv2d_readvariableop_resource+
'block30_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource*
&block20_conv2d_readvariableop_resource+
'block20_biasadd_readvariableop_resource*
&block21_conv2d_readvariableop_resource+
'block21_biasadd_readvariableop_resource*
&block22_conv2d_readvariableop_resource+
'block22_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource*
&block10_conv2d_readvariableop_resource+
'block10_biasadd_readvariableop_resource
identity??block10/BiasAdd/ReadVariableOp?block10/Conv2D/ReadVariableOp?block20/BiasAdd/ReadVariableOp?block20/Conv2D/ReadVariableOp?block21/BiasAdd/ReadVariableOp?block21/Conv2D/ReadVariableOp?block22/BiasAdd/ReadVariableOp?block22/Conv2D/ReadVariableOp?block30/BiasAdd/ReadVariableOp?block30/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
block30/Conv2D/ReadVariableOpReadVariableOp&block30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block30/Conv2D/ReadVariableOp?
block30/Conv2DConv2Dinputs%block30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block30/Conv2D?
block30/BiasAdd/ReadVariableOpReadVariableOp'block30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block30/BiasAdd/ReadVariableOp?
block30/BiasAddBiasAddblock30/Conv2D:output:0&block30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block30/BiasAddy
block30/ReluRelublock30/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block30/Reluz
conv2d_transpose/ShapeShapeblock30/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack?
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1?
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0block30/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose/BiasAdd?
block20/Conv2D/ReadVariableOpReadVariableOp&block20_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block20/Conv2D/ReadVariableOp?
block20/Conv2DConv2D!conv2d_transpose/BiasAdd:output:0%block20/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block20/Conv2D?
block20/BiasAdd/ReadVariableOpReadVariableOp'block20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block20/BiasAdd/ReadVariableOp?
block20/BiasAddBiasAddblock20/Conv2D:output:0&block20/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block20/BiasAdd{
block20/ReluRelublock20/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block20/Relu?
block21/Conv2D/ReadVariableOpReadVariableOp&block21_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block21/Conv2D/ReadVariableOp?
block21/Conv2DConv2Dblock20/Relu:activations:0%block21/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block21/Conv2D?
block21/BiasAdd/ReadVariableOpReadVariableOp'block21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block21/BiasAdd/ReadVariableOp?
block21/BiasAddBiasAddblock21/Conv2D:output:0&block21/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block21/BiasAdd{
block21/ReluRelublock21/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block21/Relu?
block22/Conv2D/ReadVariableOpReadVariableOp&block22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block22/Conv2D/ReadVariableOp?
block22/Conv2DConv2Dblock21/Relu:activations:0%block22/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block22/Conv2D?
block22/BiasAdd/ReadVariableOpReadVariableOp'block22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block22/BiasAdd/ReadVariableOp?
block22/BiasAddBiasAddblock22/Conv2D:output:0&block22/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block22/BiasAdd{
block22/ReluRelublock22/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block22/Relu~
conv2d_transpose_1/ShapeShapeblock22/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack?
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1?
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0block22/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_1/BiasAdd?
block10/Conv2D/ReadVariableOpReadVariableOp&block10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
block10/Conv2D/ReadVariableOp?
block10/Conv2DConv2D#conv2d_transpose_1/BiasAdd:output:0%block10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block10/Conv2D?
block10/BiasAdd/ReadVariableOpReadVariableOp'block10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
block10/BiasAdd/ReadVariableOp?
block10/BiasAddBiasAddblock10/Conv2D:output:0&block10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block10/BiasAddz
block10/ReluRelublock10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block10/Relu?
IdentityIdentityblock10/Relu:activations:0^block10/BiasAdd/ReadVariableOp^block10/Conv2D/ReadVariableOp^block20/BiasAdd/ReadVariableOp^block20/Conv2D/ReadVariableOp^block21/BiasAdd/ReadVariableOp^block21/Conv2D/ReadVariableOp^block22/BiasAdd/ReadVariableOp^block22/Conv2D/ReadVariableOp^block30/BiasAdd/ReadVariableOp^block30/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2@
block10/BiasAdd/ReadVariableOpblock10/BiasAdd/ReadVariableOp2>
block10/Conv2D/ReadVariableOpblock10/Conv2D/ReadVariableOp2@
block20/BiasAdd/ReadVariableOpblock20/BiasAdd/ReadVariableOp2>
block20/Conv2D/ReadVariableOpblock20/Conv2D/ReadVariableOp2@
block21/BiasAdd/ReadVariableOpblock21/BiasAdd/ReadVariableOp2>
block21/Conv2D/ReadVariableOpblock21/Conv2D/ReadVariableOp2@
block22/BiasAdd/ReadVariableOpblock22/BiasAdd/ReadVariableOp2>
block22/Conv2D/ReadVariableOpblock22/Conv2D/ReadVariableOp2@
block30/BiasAdd/ReadVariableOpblock30/BiasAdd/ReadVariableOp2>
block30/Conv2D/ReadVariableOpblock30/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_43129

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
:??*
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
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_42896

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43472

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?'
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43001
input_1/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_428422&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_428632&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_428772
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_428962&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_429172&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_429312
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_429502&
$block3_conv1/StatefulPartitionedCall?
IdentityIdentity-block3_conv1/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?B
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44493

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*B
_output_shapes0
.:,????????????????????????????*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
B__inference_block20_layer_call_and_return_conditional_losses_43150

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
1__inference_original_VGG19_B3_layer_call_fn_44645

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_435182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_block3__net_layer_call_fn_44370
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3__net_layer_call_and_return_conditional_losses_437852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameinput_tensor
?
?
#__inference_signature_wrapper_43941
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:???????????*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_428292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
'__inference_block22_layer_call_fn_43200

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block22_layer_call_and_return_conditional_losses_431922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?!
 __inference__wrapped_model_42829
input_1M
Iblock3__net_original_vgg19_b3_block1_conv1_conv2d_readvariableop_resourceN
Jblock3__net_original_vgg19_b3_block1_conv1_biasadd_readvariableop_resourceM
Iblock3__net_original_vgg19_b3_block1_conv2_conv2d_readvariableop_resourceN
Jblock3__net_original_vgg19_b3_block1_conv2_biasadd_readvariableop_resourceM
Iblock3__net_original_vgg19_b3_block2_conv1_conv2d_readvariableop_resourceN
Jblock3__net_original_vgg19_b3_block2_conv1_biasadd_readvariableop_resourceM
Iblock3__net_original_vgg19_b3_block2_conv2_conv2d_readvariableop_resourceN
Jblock3__net_original_vgg19_b3_block2_conv2_biasadd_readvariableop_resourceM
Iblock3__net_original_vgg19_b3_block3_conv1_conv2d_readvariableop_resourceN
Jblock3__net_original_vgg19_b3_block3_conv1_biasadd_readvariableop_resourceA
=block3__net_sequential_block30_conv2d_readvariableop_resourceB
>block3__net_sequential_block30_biasadd_readvariableop_resourceT
Pblock3__net_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resourceK
Gblock3__net_sequential_conv2d_transpose_biasadd_readvariableop_resourceA
=block3__net_sequential_block20_conv2d_readvariableop_resourceB
>block3__net_sequential_block20_biasadd_readvariableop_resourceA
=block3__net_sequential_block21_conv2d_readvariableop_resourceB
>block3__net_sequential_block21_biasadd_readvariableop_resourceA
=block3__net_sequential_block22_conv2d_readvariableop_resourceB
>block3__net_sequential_block22_biasadd_readvariableop_resourceV
Rblock3__net_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceM
Iblock3__net_sequential_conv2d_transpose_1_biasadd_readvariableop_resourceA
=block3__net_sequential_block10_conv2d_readvariableop_resourceB
>block3__net_sequential_block10_biasadd_readvariableop_resource7
3block3__net_block_d1_conv2d_readvariableop_resource8
4block3__net_block_d1_biasadd_readvariableop_resource
identity??+block3__net/Block_D1/BiasAdd/ReadVariableOp?*block3__net/Block_D1/Conv2D/ReadVariableOp?Ablock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?@block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?Ablock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?@block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?Ablock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?@block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?Ablock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?@block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?Ablock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?@block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?Cblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?Bblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?Cblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?Bblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?Cblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?Bblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?Cblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?Bblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?Cblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?Bblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?5block3__net/sequential/block10/BiasAdd/ReadVariableOp?4block3__net/sequential/block10/Conv2D/ReadVariableOp?5block3__net/sequential/block20/BiasAdd/ReadVariableOp?4block3__net/sequential/block20/Conv2D/ReadVariableOp?5block3__net/sequential/block21/BiasAdd/ReadVariableOp?4block3__net/sequential/block21/Conv2D/ReadVariableOp?5block3__net/sequential/block22/BiasAdd/ReadVariableOp?4block3__net/sequential/block22/Conv2D/ReadVariableOp?5block3__net/sequential/block30/BiasAdd/ReadVariableOp?4block3__net/sequential/block30/Conv2D/ReadVariableOp?>block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOp?Gblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?@block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?Iblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
8block3__net/original_VGG19_B3/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8block3__net/original_VGG19_B3/block1_conv1/dilation_rate?
@block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02B
@block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?
1block3__net/original_VGG19_B3/block1_conv1/Conv2DConv2Dinput_1Hblock3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
23
1block3__net/original_VGG19_B3/block1_conv1/Conv2D?
Ablock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ablock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?
2block3__net/original_VGG19_B3/block1_conv1/BiasAddBiasAdd:block3__net/original_VGG19_B3/block1_conv1/Conv2D:output:0Iblock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@24
2block3__net/original_VGG19_B3/block1_conv1/BiasAdd?
/block3__net/original_VGG19_B3/block1_conv1/ReluRelu;block3__net/original_VGG19_B3/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@21
/block3__net/original_VGG19_B3/block1_conv1/Relu?
8block3__net/original_VGG19_B3/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8block3__net/original_VGG19_B3/block1_conv2/dilation_rate?
@block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?
1block3__net/original_VGG19_B3/block1_conv2/Conv2DConv2D=block3__net/original_VGG19_B3/block1_conv1/Relu:activations:0Hblock3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
23
1block3__net/original_VGG19_B3/block1_conv2/Conv2D?
Ablock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ablock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?
2block3__net/original_VGG19_B3/block1_conv2/BiasAddBiasAdd:block3__net/original_VGG19_B3/block1_conv2/Conv2D:output:0Iblock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@24
2block3__net/original_VGG19_B3/block1_conv2/BiasAdd?
/block3__net/original_VGG19_B3/block1_conv2/ReluRelu;block3__net/original_VGG19_B3/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@21
/block3__net/original_VGG19_B3/block1_conv2/Relu?
1block3__net/original_VGG19_B3/block1_pool/MaxPoolMaxPool=block3__net/original_VGG19_B3/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
23
1block3__net/original_VGG19_B3/block1_pool/MaxPool?
8block3__net/original_VGG19_B3/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8block3__net/original_VGG19_B3/block2_conv1/dilation_rate?
@block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?
1block3__net/original_VGG19_B3/block2_conv1/Conv2DConv2D:block3__net/original_VGG19_B3/block1_pool/MaxPool:output:0Hblock3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
23
1block3__net/original_VGG19_B3/block2_conv1/Conv2D?
Ablock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Ablock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?
2block3__net/original_VGG19_B3/block2_conv1/BiasAddBiasAdd:block3__net/original_VGG19_B3/block2_conv1/Conv2D:output:0Iblock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????24
2block3__net/original_VGG19_B3/block2_conv1/BiasAdd?
/block3__net/original_VGG19_B3/block2_conv1/ReluRelu;block3__net/original_VGG19_B3/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????21
/block3__net/original_VGG19_B3/block2_conv1/Relu?
8block3__net/original_VGG19_B3/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8block3__net/original_VGG19_B3/block2_conv2/dilation_rate?
@block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?
1block3__net/original_VGG19_B3/block2_conv2/Conv2DConv2D=block3__net/original_VGG19_B3/block2_conv1/Relu:activations:0Hblock3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
23
1block3__net/original_VGG19_B3/block2_conv2/Conv2D?
Ablock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Ablock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?
2block3__net/original_VGG19_B3/block2_conv2/BiasAddBiasAdd:block3__net/original_VGG19_B3/block2_conv2/Conv2D:output:0Iblock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????24
2block3__net/original_VGG19_B3/block2_conv2/BiasAdd?
/block3__net/original_VGG19_B3/block2_conv2/ReluRelu;block3__net/original_VGG19_B3/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????21
/block3__net/original_VGG19_B3/block2_conv2/Relu?
1block3__net/original_VGG19_B3/block2_pool/MaxPoolMaxPool=block3__net/original_VGG19_B3/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
23
1block3__net/original_VGG19_B3/block2_pool/MaxPool?
8block3__net/original_VGG19_B3/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8block3__net/original_VGG19_B3/block3_conv1/dilation_rate?
@block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?
1block3__net/original_VGG19_B3/block3_conv1/Conv2DConv2D:block3__net/original_VGG19_B3/block2_pool/MaxPool:output:0Hblock3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
23
1block3__net/original_VGG19_B3/block3_conv1/Conv2D?
Ablock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Ablock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?
2block3__net/original_VGG19_B3/block3_conv1/BiasAddBiasAdd:block3__net/original_VGG19_B3/block3_conv1/Conv2D:output:0Iblock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?24
2block3__net/original_VGG19_B3/block3_conv1/BiasAdd?
/block3__net/original_VGG19_B3/block3_conv1/ReluRelu;block3__net/original_VGG19_B3/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?21
/block3__net/original_VGG19_B3/block3_conv1/Relu?
4block3__net/sequential/block30/Conv2D/ReadVariableOpReadVariableOp=block3__net_sequential_block30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4block3__net/sequential/block30/Conv2D/ReadVariableOp?
%block3__net/sequential/block30/Conv2DConv2D=block3__net/original_VGG19_B3/block3_conv1/Relu:activations:0<block3__net/sequential/block30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2'
%block3__net/sequential/block30/Conv2D?
5block3__net/sequential/block30/BiasAdd/ReadVariableOpReadVariableOp>block3__net_sequential_block30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5block3__net/sequential/block30/BiasAdd/ReadVariableOp?
&block3__net/sequential/block30/BiasAddBiasAdd.block3__net/sequential/block30/Conv2D:output:0=block3__net/sequential/block30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2(
&block3__net/sequential/block30/BiasAdd?
#block3__net/sequential/block30/ReluRelu/block3__net/sequential/block30/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2%
#block3__net/sequential/block30/Relu?
-block3__net/sequential/conv2d_transpose/ShapeShape1block3__net/sequential/block30/Relu:activations:0*
T0*
_output_shapes
:2/
-block3__net/sequential/conv2d_transpose/Shape?
;block3__net/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;block3__net/sequential/conv2d_transpose/strided_slice/stack?
=block3__net/sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=block3__net/sequential/conv2d_transpose/strided_slice/stack_1?
=block3__net/sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=block3__net/sequential/conv2d_transpose/strided_slice/stack_2?
5block3__net/sequential/conv2d_transpose/strided_sliceStridedSlice6block3__net/sequential/conv2d_transpose/Shape:output:0Dblock3__net/sequential/conv2d_transpose/strided_slice/stack:output:0Fblock3__net/sequential/conv2d_transpose/strided_slice/stack_1:output:0Fblock3__net/sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5block3__net/sequential/conv2d_transpose/strided_slice?
=block3__net/sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=block3__net/sequential/conv2d_transpose/strided_slice_1/stack?
?block3__net/sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_1/stack_1?
?block3__net/sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_1/stack_2?
7block3__net/sequential/conv2d_transpose/strided_slice_1StridedSlice6block3__net/sequential/conv2d_transpose/Shape:output:0Fblock3__net/sequential/conv2d_transpose/strided_slice_1/stack:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_1/stack_1:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7block3__net/sequential/conv2d_transpose/strided_slice_1?
=block3__net/sequential/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=block3__net/sequential/conv2d_transpose/strided_slice_2/stack?
?block3__net/sequential/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_2/stack_1?
?block3__net/sequential/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_2/stack_2?
7block3__net/sequential/conv2d_transpose/strided_slice_2StridedSlice6block3__net/sequential/conv2d_transpose/Shape:output:0Fblock3__net/sequential/conv2d_transpose/strided_slice_2/stack:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_2/stack_1:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7block3__net/sequential/conv2d_transpose/strided_slice_2?
-block3__net/sequential/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-block3__net/sequential/conv2d_transpose/mul/y?
+block3__net/sequential/conv2d_transpose/mulMul@block3__net/sequential/conv2d_transpose/strided_slice_1:output:06block3__net/sequential/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2-
+block3__net/sequential/conv2d_transpose/mul?
/block3__net/sequential/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :21
/block3__net/sequential/conv2d_transpose/mul_1/y?
-block3__net/sequential/conv2d_transpose/mul_1Mul@block3__net/sequential/conv2d_transpose/strided_slice_2:output:08block3__net/sequential/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2/
-block3__net/sequential/conv2d_transpose/mul_1?
/block3__net/sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?21
/block3__net/sequential/conv2d_transpose/stack/3?
-block3__net/sequential/conv2d_transpose/stackPack>block3__net/sequential/conv2d_transpose/strided_slice:output:0/block3__net/sequential/conv2d_transpose/mul:z:01block3__net/sequential/conv2d_transpose/mul_1:z:08block3__net/sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-block3__net/sequential/conv2d_transpose/stack?
=block3__net/sequential/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=block3__net/sequential/conv2d_transpose/strided_slice_3/stack?
?block3__net/sequential/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_3/stack_1?
?block3__net/sequential/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose/strided_slice_3/stack_2?
7block3__net/sequential/conv2d_transpose/strided_slice_3StridedSlice6block3__net/sequential/conv2d_transpose/stack:output:0Fblock3__net/sequential/conv2d_transpose/strided_slice_3/stack:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_3/stack_1:output:0Hblock3__net/sequential/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7block3__net/sequential/conv2d_transpose/strided_slice_3?
Gblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpPblock3__net_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02I
Gblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?
8block3__net/sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput6block3__net/sequential/conv2d_transpose/stack:output:0Oblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:01block3__net/sequential/block30/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2:
8block3__net/sequential/conv2d_transpose/conv2d_transpose?
>block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpGblock3__net_sequential_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOp?
/block3__net/sequential/conv2d_transpose/BiasAddBiasAddAblock3__net/sequential/conv2d_transpose/conv2d_transpose:output:0Fblock3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????21
/block3__net/sequential/conv2d_transpose/BiasAdd?
4block3__net/sequential/block20/Conv2D/ReadVariableOpReadVariableOp=block3__net_sequential_block20_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4block3__net/sequential/block20/Conv2D/ReadVariableOp?
%block3__net/sequential/block20/Conv2DConv2D8block3__net/sequential/conv2d_transpose/BiasAdd:output:0<block3__net/sequential/block20/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%block3__net/sequential/block20/Conv2D?
5block3__net/sequential/block20/BiasAdd/ReadVariableOpReadVariableOp>block3__net_sequential_block20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5block3__net/sequential/block20/BiasAdd/ReadVariableOp?
&block3__net/sequential/block20/BiasAddBiasAdd.block3__net/sequential/block20/Conv2D:output:0=block3__net/sequential/block20/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&block3__net/sequential/block20/BiasAdd?
#block3__net/sequential/block20/ReluRelu/block3__net/sequential/block20/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#block3__net/sequential/block20/Relu?
4block3__net/sequential/block21/Conv2D/ReadVariableOpReadVariableOp=block3__net_sequential_block21_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4block3__net/sequential/block21/Conv2D/ReadVariableOp?
%block3__net/sequential/block21/Conv2DConv2D1block3__net/sequential/block20/Relu:activations:0<block3__net/sequential/block21/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%block3__net/sequential/block21/Conv2D?
5block3__net/sequential/block21/BiasAdd/ReadVariableOpReadVariableOp>block3__net_sequential_block21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5block3__net/sequential/block21/BiasAdd/ReadVariableOp?
&block3__net/sequential/block21/BiasAddBiasAdd.block3__net/sequential/block21/Conv2D:output:0=block3__net/sequential/block21/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&block3__net/sequential/block21/BiasAdd?
#block3__net/sequential/block21/ReluRelu/block3__net/sequential/block21/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#block3__net/sequential/block21/Relu?
4block3__net/sequential/block22/Conv2D/ReadVariableOpReadVariableOp=block3__net_sequential_block22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4block3__net/sequential/block22/Conv2D/ReadVariableOp?
%block3__net/sequential/block22/Conv2DConv2D1block3__net/sequential/block21/Relu:activations:0<block3__net/sequential/block22/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%block3__net/sequential/block22/Conv2D?
5block3__net/sequential/block22/BiasAdd/ReadVariableOpReadVariableOp>block3__net_sequential_block22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5block3__net/sequential/block22/BiasAdd/ReadVariableOp?
&block3__net/sequential/block22/BiasAddBiasAdd.block3__net/sequential/block22/Conv2D:output:0=block3__net/sequential/block22/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&block3__net/sequential/block22/BiasAdd?
#block3__net/sequential/block22/ReluRelu/block3__net/sequential/block22/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#block3__net/sequential/block22/Relu?
/block3__net/sequential/conv2d_transpose_1/ShapeShape1block3__net/sequential/block22/Relu:activations:0*
T0*
_output_shapes
:21
/block3__net/sequential/conv2d_transpose_1/Shape?
=block3__net/sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=block3__net/sequential/conv2d_transpose_1/strided_slice/stack?
?block3__net/sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose_1/strided_slice/stack_1?
?block3__net/sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose_1/strided_slice/stack_2?
7block3__net/sequential/conv2d_transpose_1/strided_sliceStridedSlice8block3__net/sequential/conv2d_transpose_1/Shape:output:0Fblock3__net/sequential/conv2d_transpose_1/strided_slice/stack:output:0Hblock3__net/sequential/conv2d_transpose_1/strided_slice/stack_1:output:0Hblock3__net/sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7block3__net/sequential/conv2d_transpose_1/strided_slice?
?block3__net/sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose_1/strided_slice_1/stack?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_1?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_2?
9block3__net/sequential/conv2d_transpose_1/strided_slice_1StridedSlice8block3__net/sequential/conv2d_transpose_1/Shape:output:0Hblock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9block3__net/sequential/conv2d_transpose_1/strided_slice_1?
?block3__net/sequential/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?block3__net/sequential/conv2d_transpose_1/strided_slice_2/stack?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_1?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_2?
9block3__net/sequential/conv2d_transpose_1/strided_slice_2StridedSlice8block3__net/sequential/conv2d_transpose_1/Shape:output:0Hblock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_1:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9block3__net/sequential/conv2d_transpose_1/strided_slice_2?
/block3__net/sequential/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :21
/block3__net/sequential/conv2d_transpose_1/mul/y?
-block3__net/sequential/conv2d_transpose_1/mulMulBblock3__net/sequential/conv2d_transpose_1/strided_slice_1:output:08block3__net/sequential/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2/
-block3__net/sequential/conv2d_transpose_1/mul?
1block3__net/sequential/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :23
1block3__net/sequential/conv2d_transpose_1/mul_1/y?
/block3__net/sequential/conv2d_transpose_1/mul_1MulBblock3__net/sequential/conv2d_transpose_1/strided_slice_2:output:0:block3__net/sequential/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 21
/block3__net/sequential/conv2d_transpose_1/mul_1?
1block3__net/sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@23
1block3__net/sequential/conv2d_transpose_1/stack/3?
/block3__net/sequential/conv2d_transpose_1/stackPack@block3__net/sequential/conv2d_transpose_1/strided_slice:output:01block3__net/sequential/conv2d_transpose_1/mul:z:03block3__net/sequential/conv2d_transpose_1/mul_1:z:0:block3__net/sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:21
/block3__net/sequential/conv2d_transpose_1/stack?
?block3__net/sequential/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?block3__net/sequential/conv2d_transpose_1/strided_slice_3/stack?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_1?
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Ablock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_2?
9block3__net/sequential/conv2d_transpose_1/strided_slice_3StridedSlice8block3__net/sequential/conv2d_transpose_1/stack:output:0Hblock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_1:output:0Jblock3__net/sequential/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9block3__net/sequential/conv2d_transpose_1/strided_slice_3?
Iblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpRblock3__net_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02K
Iblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
:block3__net/sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput8block3__net/sequential/conv2d_transpose_1/stack:output:0Qblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:01block3__net/sequential/block22/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2<
:block3__net/sequential/conv2d_transpose_1/conv2d_transpose?
@block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpIblock3__net_sequential_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?
1block3__net/sequential/conv2d_transpose_1/BiasAddBiasAddCblock3__net/sequential/conv2d_transpose_1/conv2d_transpose:output:0Hblock3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@23
1block3__net/sequential/conv2d_transpose_1/BiasAdd?
4block3__net/sequential/block10/Conv2D/ReadVariableOpReadVariableOp=block3__net_sequential_block10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype026
4block3__net/sequential/block10/Conv2D/ReadVariableOp?
%block3__net/sequential/block10/Conv2DConv2D:block3__net/sequential/conv2d_transpose_1/BiasAdd:output:0<block3__net/sequential/block10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%block3__net/sequential/block10/Conv2D?
5block3__net/sequential/block10/BiasAdd/ReadVariableOpReadVariableOp>block3__net_sequential_block10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5block3__net/sequential/block10/BiasAdd/ReadVariableOp?
&block3__net/sequential/block10/BiasAddBiasAdd.block3__net/sequential/block10/Conv2D:output:0=block3__net/sequential/block10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&block3__net/sequential/block10/BiasAdd?
#block3__net/sequential/block10/ReluRelu/block3__net/sequential/block10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#block3__net/sequential/block10/Relu?
*block3__net/Block_D1/Conv2D/ReadVariableOpReadVariableOp3block3__net_block_d1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*block3__net/Block_D1/Conv2D/ReadVariableOp?
block3__net/Block_D1/Conv2DConv2D1block3__net/sequential/block10/Relu:activations:02block3__net/Block_D1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
block3__net/Block_D1/Conv2D?
+block3__net/Block_D1/BiasAdd/ReadVariableOpReadVariableOp4block3__net_block_d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+block3__net/Block_D1/BiasAdd/ReadVariableOp?
block3__net/Block_D1/BiasAddBiasAdd$block3__net/Block_D1/Conv2D:output:03block3__net/Block_D1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
block3__net/Block_D1/BiasAdd?
:block3__net/original_VGG19_B3_1/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:block3__net/original_VGG19_B3_1/block1_conv1/dilation_rate?
Bblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block1_conv1_conv2d_readvariableop_resourceA^block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype02D
Bblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?
3block3__net/original_VGG19_B3_1/block1_conv1/Conv2DConv2D%block3__net/Block_D1/BiasAdd:output:0Jblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
25
3block3__net/original_VGG19_B3_1/block1_conv1/Conv2D?
Cblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block1_conv1_biasadd_readvariableop_resourceB^block3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype02E
Cblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?
4block3__net/original_VGG19_B3_1/block1_conv1/BiasAddBiasAdd<block3__net/original_VGG19_B3_1/block1_conv1/Conv2D:output:0Kblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@26
4block3__net/original_VGG19_B3_1/block1_conv1/BiasAdd?
1block3__net/original_VGG19_B3_1/block1_conv1/ReluRelu=block3__net/original_VGG19_B3_1/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@23
1block3__net/original_VGG19_B3_1/block1_conv1/Relu?
:block3__net/original_VGG19_B3_1/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:block3__net/original_VGG19_B3_1/block1_conv2/dilation_rate?
Bblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block1_conv2_conv2d_readvariableop_resourceA^block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02D
Bblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?
3block3__net/original_VGG19_B3_1/block1_conv2/Conv2DConv2D?block3__net/original_VGG19_B3_1/block1_conv1/Relu:activations:0Jblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
25
3block3__net/original_VGG19_B3_1/block1_conv2/Conv2D?
Cblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block1_conv2_biasadd_readvariableop_resourceB^block3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype02E
Cblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?
4block3__net/original_VGG19_B3_1/block1_conv2/BiasAddBiasAdd<block3__net/original_VGG19_B3_1/block1_conv2/Conv2D:output:0Kblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@26
4block3__net/original_VGG19_B3_1/block1_conv2/BiasAdd?
1block3__net/original_VGG19_B3_1/block1_conv2/ReluRelu=block3__net/original_VGG19_B3_1/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@23
1block3__net/original_VGG19_B3_1/block1_conv2/Relu?
3block3__net/original_VGG19_B3_1/block1_pool/MaxPoolMaxPool?block3__net/original_VGG19_B3_1/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
25
3block3__net/original_VGG19_B3_1/block1_pool/MaxPool?
:block3__net/original_VGG19_B3_1/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:block3__net/original_VGG19_B3_1/block2_conv1/dilation_rate?
Bblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block2_conv1_conv2d_readvariableop_resourceA^block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp*'
_output_shapes
:@?*
dtype02D
Bblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?
3block3__net/original_VGG19_B3_1/block2_conv1/Conv2DConv2D<block3__net/original_VGG19_B3_1/block1_pool/MaxPool:output:0Jblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
25
3block3__net/original_VGG19_B3_1/block2_conv1/Conv2D?
Cblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block2_conv1_biasadd_readvariableop_resourceB^block3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype02E
Cblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?
4block3__net/original_VGG19_B3_1/block2_conv1/BiasAddBiasAdd<block3__net/original_VGG19_B3_1/block2_conv1/Conv2D:output:0Kblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????26
4block3__net/original_VGG19_B3_1/block2_conv1/BiasAdd?
1block3__net/original_VGG19_B3_1/block2_conv1/ReluRelu=block3__net/original_VGG19_B3_1/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????23
1block3__net/original_VGG19_B3_1/block2_conv1/Relu?
:block3__net/original_VGG19_B3_1/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:block3__net/original_VGG19_B3_1/block2_conv2/dilation_rate?
Bblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block2_conv2_conv2d_readvariableop_resourceA^block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype02D
Bblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?
3block3__net/original_VGG19_B3_1/block2_conv2/Conv2DConv2D?block3__net/original_VGG19_B3_1/block2_conv1/Relu:activations:0Jblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
25
3block3__net/original_VGG19_B3_1/block2_conv2/Conv2D?
Cblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block2_conv2_biasadd_readvariableop_resourceB^block3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype02E
Cblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?
4block3__net/original_VGG19_B3_1/block2_conv2/BiasAddBiasAdd<block3__net/original_VGG19_B3_1/block2_conv2/Conv2D:output:0Kblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????26
4block3__net/original_VGG19_B3_1/block2_conv2/BiasAdd?
1block3__net/original_VGG19_B3_1/block2_conv2/ReluRelu=block3__net/original_VGG19_B3_1/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????23
1block3__net/original_VGG19_B3_1/block2_conv2/Relu?
3block3__net/original_VGG19_B3_1/block2_pool/MaxPoolMaxPool?block3__net/original_VGG19_B3_1/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
25
3block3__net/original_VGG19_B3_1/block2_pool/MaxPool?
:block3__net/original_VGG19_B3_1/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:block3__net/original_VGG19_B3_1/block3_conv1/dilation_rate?
Bblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOpReadVariableOpIblock3__net_original_vgg19_b3_block3_conv1_conv2d_readvariableop_resourceA^block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype02D
Bblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?
3block3__net/original_VGG19_B3_1/block3_conv1/Conv2DConv2D<block3__net/original_VGG19_B3_1/block2_pool/MaxPool:output:0Jblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
25
3block3__net/original_VGG19_B3_1/block3_conv1/Conv2D?
Cblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpJblock3__net_original_vgg19_b3_block3_conv1_biasadd_readvariableop_resourceB^block3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype02E
Cblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?
4block3__net/original_VGG19_B3_1/block3_conv1/BiasAddBiasAdd<block3__net/original_VGG19_B3_1/block3_conv1/Conv2D:output:0Kblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?26
4block3__net/original_VGG19_B3_1/block3_conv1/BiasAdd?
1block3__net/original_VGG19_B3_1/block3_conv1/ReluRelu=block3__net/original_VGG19_B3_1/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?23
1block3__net/original_VGG19_B3_1/block3_conv1/Relu?
0block3__net/mean_squared_error/SquaredDifferenceSquaredDifference?block3__net/original_VGG19_B3_1/block3_conv1/Relu:activations:0=block3__net/original_VGG19_B3/block3_conv1/Relu:activations:0*
T0*0
_output_shapes
:?????????``?22
0block3__net/mean_squared_error/SquaredDifference?
5block3__net/mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5block3__net/mean_squared_error/Mean/reduction_indices?
#block3__net/mean_squared_error/MeanMean4block3__net/mean_squared_error/SquaredDifference:z:0>block3__net/mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2%
#block3__net/mean_squared_error/Mean?
3block3__net/mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3block3__net/mean_squared_error/weighted_loss/Cast/x?
ablock3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2c
ablock3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
`block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2b
`block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
`block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape,block3__net/mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2b
`block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
_block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2a
_block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
oblock3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2q
oblock3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Nblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape,block3__net/mean_squared_error/Mean:output:0p^block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2P
Nblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Nblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstp^block3__net/mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2P
Nblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
Hblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillWblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Wblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2J
Hblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like?
>block3__net/mean_squared_error/weighted_loss/broadcast_weightsMul<block3__net/mean_squared_error/weighted_loss/Cast/x:output:0Qblock3__net/mean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``2@
>block3__net/mean_squared_error/weighted_loss/broadcast_weights?
0block3__net/mean_squared_error/weighted_loss/MulMul,block3__net/mean_squared_error/Mean:output:0Bblock3__net/mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``22
0block3__net/mean_squared_error/weighted_loss/Mul?
2block3__net/mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          24
2block3__net/mean_squared_error/weighted_loss/Const?
0block3__net/mean_squared_error/weighted_loss/SumSum4block3__net/mean_squared_error/weighted_loss/Mul:z:0;block3__net/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 22
0block3__net/mean_squared_error/weighted_loss/Sum?
9block3__net/mean_squared_error/weighted_loss/num_elementsSize4block3__net/mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2;
9block3__net/mean_squared_error/weighted_loss/num_elements?
>block3__net/mean_squared_error/weighted_loss/num_elements/CastCastBblock3__net/mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>block3__net/mean_squared_error/weighted_loss/num_elements/Cast?
4block3__net/mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 26
4block3__net/mean_squared_error/weighted_loss/Const_1?
2block3__net/mean_squared_error/weighted_loss/Sum_1Sum9block3__net/mean_squared_error/weighted_loss/Sum:output:0=block3__net/mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 24
2block3__net/mean_squared_error/weighted_loss/Sum_1?
2block3__net/mean_squared_error/weighted_loss/valueDivNoNan;block3__net/mean_squared_error/weighted_loss/Sum_1:output:0Bblock3__net/mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 24
2block3__net/mean_squared_error/weighted_loss/valuek
block3__net/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
block3__net/mul/x?
block3__net/mulMulblock3__net/mul/x:output:06block3__net/mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
block3__net/mul?
IdentityIdentity%block3__net/Block_D1/BiasAdd:output:0,^block3__net/Block_D1/BiasAdd/ReadVariableOp+^block3__net/Block_D1/Conv2D/ReadVariableOpB^block3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOpA^block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOpB^block3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOpA^block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOpB^block3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOpA^block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOpB^block3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOpA^block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOpB^block3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOpA^block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOpD^block3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOpC^block3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOpD^block3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOpC^block3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOpD^block3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOpC^block3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOpD^block3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOpC^block3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOpD^block3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOpC^block3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp6^block3__net/sequential/block10/BiasAdd/ReadVariableOp5^block3__net/sequential/block10/Conv2D/ReadVariableOp6^block3__net/sequential/block20/BiasAdd/ReadVariableOp5^block3__net/sequential/block20/Conv2D/ReadVariableOp6^block3__net/sequential/block21/BiasAdd/ReadVariableOp5^block3__net/sequential/block21/Conv2D/ReadVariableOp6^block3__net/sequential/block22/BiasAdd/ReadVariableOp5^block3__net/sequential/block22/Conv2D/ReadVariableOp6^block3__net/sequential/block30/BiasAdd/ReadVariableOp5^block3__net/sequential/block30/Conv2D/ReadVariableOp?^block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOpH^block3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpA^block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpJ^block3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2Z
+block3__net/Block_D1/BiasAdd/ReadVariableOp+block3__net/Block_D1/BiasAdd/ReadVariableOp2X
*block3__net/Block_D1/Conv2D/ReadVariableOp*block3__net/Block_D1/Conv2D/ReadVariableOp2?
Ablock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOpAblock3__net/original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp2?
@block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp@block3__net/original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp2?
Ablock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOpAblock3__net/original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp2?
@block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp@block3__net/original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp2?
Ablock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOpAblock3__net/original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp2?
@block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp@block3__net/original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp2?
Ablock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOpAblock3__net/original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp2?
@block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp@block3__net/original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp2?
Ablock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOpAblock3__net/original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp2?
@block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp@block3__net/original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp2?
Cblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOpCblock3__net/original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp2?
Bblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOpBblock3__net/original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp2?
Cblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOpCblock3__net/original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp2?
Bblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOpBblock3__net/original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp2?
Cblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOpCblock3__net/original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp2?
Bblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOpBblock3__net/original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp2?
Cblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOpCblock3__net/original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp2?
Bblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOpBblock3__net/original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp2?
Cblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOpCblock3__net/original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp2?
Bblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOpBblock3__net/original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp2n
5block3__net/sequential/block10/BiasAdd/ReadVariableOp5block3__net/sequential/block10/BiasAdd/ReadVariableOp2l
4block3__net/sequential/block10/Conv2D/ReadVariableOp4block3__net/sequential/block10/Conv2D/ReadVariableOp2n
5block3__net/sequential/block20/BiasAdd/ReadVariableOp5block3__net/sequential/block20/BiasAdd/ReadVariableOp2l
4block3__net/sequential/block20/Conv2D/ReadVariableOp4block3__net/sequential/block20/Conv2D/ReadVariableOp2n
5block3__net/sequential/block21/BiasAdd/ReadVariableOp5block3__net/sequential/block21/BiasAdd/ReadVariableOp2l
4block3__net/sequential/block21/Conv2D/ReadVariableOp4block3__net/sequential/block21/Conv2D/ReadVariableOp2n
5block3__net/sequential/block22/BiasAdd/ReadVariableOp5block3__net/sequential/block22/BiasAdd/ReadVariableOp2l
4block3__net/sequential/block22/Conv2D/ReadVariableOp4block3__net/sequential/block22/Conv2D/ReadVariableOp2n
5block3__net/sequential/block30/BiasAdd/ReadVariableOp5block3__net/sequential/block30/BiasAdd/ReadVariableOp2l
4block3__net/sequential/block30/Conv2D/ReadVariableOp4block3__net/sequential/block30/Conv2D/ReadVariableOp2?
>block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOp>block3__net/sequential/conv2d_transpose/BiasAdd/ReadVariableOp2?
Gblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpGblock3__net/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2?
@block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp@block3__net/sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Iblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpIblock3__net/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:' #
!
_user_specified_name	input_1
?
?
1__inference_original_VGG19_B3_layer_call_fn_44508

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?U
?
F__inference_block3__net_layer_call_and_return_conditional_losses_43872
input_tensor4
0original_vgg19_b3_statefulpartitionedcall_args_14
0original_vgg19_b3_statefulpartitionedcall_args_24
0original_vgg19_b3_statefulpartitionedcall_args_34
0original_vgg19_b3_statefulpartitionedcall_args_44
0original_vgg19_b3_statefulpartitionedcall_args_54
0original_vgg19_b3_statefulpartitionedcall_args_64
0original_vgg19_b3_statefulpartitionedcall_args_74
0original_vgg19_b3_statefulpartitionedcall_args_84
0original_vgg19_b3_statefulpartitionedcall_args_95
1original_vgg19_b3_statefulpartitionedcall_args_10-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10.
*sequential_statefulpartitionedcall_args_11.
*sequential_statefulpartitionedcall_args_12.
*sequential_statefulpartitionedcall_args_13.
*sequential_statefulpartitionedcall_args_14+
'block_d1_statefulpartitionedcall_args_1+
'block_d1_statefulpartitionedcall_args_2
identity?? Block_D1/StatefulPartitionedCall?)original_VGG19_B3/StatefulPartitionedCall?+original_VGG19_B3_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
)original_VGG19_B3/StatefulPartitionedCallStatefulPartitionedCallinput_tensor0original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_435182+
)original_VGG19_B3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall2original_VGG19_B3/StatefulPartitionedCall:output:0)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*sequential_statefulpartitionedcall_args_11*sequential_statefulpartitionedcall_args_12*sequential_statefulpartitionedcall_args_13*sequential_statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433862$
"sequential/StatefulPartitionedCall?
 Block_D1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0'block_d1_statefulpartitionedcall_args_1'block_d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_Block_D1_layer_call_and_return_conditional_losses_434152"
 Block_D1/StatefulPartitionedCall?
+original_VGG19_B3_1/StatefulPartitionedCallStatefulPartitionedCall)Block_D1/StatefulPartitionedCall:output:00original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*^original_VGG19_B3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430612-
+original_VGG19_B3_1/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference4original_VGG19_B3_1/StatefulPartitionedCall:output:02original_VGG19_B3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentity)Block_D1/StatefulPartitionedCall:output:0!^Block_D1/StatefulPartitionedCall*^original_VGG19_B3/StatefulPartitionedCall,^original_VGG19_B3_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2D
 Block_D1/StatefulPartitionedCall Block_D1/StatefulPartitionedCall2V
)original_VGG19_B3/StatefulPartitionedCall)original_VGG19_B3/StatefulPartitionedCall2Z
+original_VGG19_B3_1/StatefulPartitionedCall+original_VGG19_B3_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:, (
&
_user_specified_nameinput_tensor
?
?
*__inference_sequential_layer_call_fn_44877

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?+
?
E__inference_sequential_layer_call_and_return_conditional_losses_43386

inputs*
&block30_statefulpartitionedcall_args_1*
&block30_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_2*
&block20_statefulpartitionedcall_args_1*
&block20_statefulpartitionedcall_args_2*
&block21_statefulpartitionedcall_args_1*
&block21_statefulpartitionedcall_args_2*
&block22_statefulpartitionedcall_args_1*
&block22_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_2*
&block10_statefulpartitionedcall_args_1*
&block10_statefulpartitionedcall_args_2
identity??block10/StatefulPartitionedCall?block20/StatefulPartitionedCall?block21/StatefulPartitionedCall?block22/StatefulPartitionedCall?block30/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?
block30/StatefulPartitionedCallStatefulPartitionedCallinputs&block30_statefulpartitionedcall_args_1&block30_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block30_layer_call_and_return_conditional_losses_430872!
block30/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(block30/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_431292*
(conv2d_transpose/StatefulPartitionedCall?
block20/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0&block20_statefulpartitionedcall_args_1&block20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block20_layer_call_and_return_conditional_losses_431502!
block20/StatefulPartitionedCall?
block21/StatefulPartitionedCallStatefulPartitionedCall(block20/StatefulPartitionedCall:output:0&block21_statefulpartitionedcall_args_1&block21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block21_layer_call_and_return_conditional_losses_431712!
block21/StatefulPartitionedCall?
block22/StatefulPartitionedCallStatefulPartitionedCall(block21/StatefulPartitionedCall:output:0&block22_statefulpartitionedcall_args_1&block22_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block22_layer_call_and_return_conditional_losses_431922!
block22/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(block22/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_432342,
*conv2d_transpose_1/StatefulPartitionedCall?
block10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0&block10_statefulpartitionedcall_args_1&block10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block10_layer_call_and_return_conditional_losses_432552!
block10/StatefulPartitionedCall?
IdentityIdentity(block10/StatefulPartitionedCall:output:0 ^block10/StatefulPartitionedCall ^block20/StatefulPartitionedCall ^block21/StatefulPartitionedCall ^block22/StatefulPartitionedCall ^block30/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2B
block10/StatefulPartitionedCallblock10/StatefulPartitionedCall2B
block20/StatefulPartitionedCallblock20/StatefulPartitionedCall2B
block21/StatefulPartitionedCallblock21/StatefulPartitionedCall2B
block22/StatefulPartitionedCallblock22/StatefulPartitionedCall2B
block30/StatefulPartitionedCallblock30/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
'__inference_block30_layer_call_fn_43095

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block30_layer_call_and_return_conditional_losses_430872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
B__inference_block21_layer_call_and_return_conditional_losses_43171

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
,__inference_block2_conv2_layer_call_fn_42925

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_429172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
1__inference_original_VGG19_B3_layer_call_fn_43074
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?+
?
E__inference_sequential_layer_call_and_return_conditional_losses_43289
input_1*
&block30_statefulpartitionedcall_args_1*
&block30_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_2*
&block20_statefulpartitionedcall_args_1*
&block20_statefulpartitionedcall_args_2*
&block21_statefulpartitionedcall_args_1*
&block21_statefulpartitionedcall_args_2*
&block22_statefulpartitionedcall_args_1*
&block22_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_2*
&block10_statefulpartitionedcall_args_1*
&block10_statefulpartitionedcall_args_2
identity??block10/StatefulPartitionedCall?block20/StatefulPartitionedCall?block21/StatefulPartitionedCall?block22/StatefulPartitionedCall?block30/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?
block30/StatefulPartitionedCallStatefulPartitionedCallinput_1&block30_statefulpartitionedcall_args_1&block30_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block30_layer_call_and_return_conditional_losses_430872!
block30/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(block30/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_431292*
(conv2d_transpose/StatefulPartitionedCall?
block20/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0&block20_statefulpartitionedcall_args_1&block20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block20_layer_call_and_return_conditional_losses_431502!
block20/StatefulPartitionedCall?
block21/StatefulPartitionedCallStatefulPartitionedCall(block20/StatefulPartitionedCall:output:0&block21_statefulpartitionedcall_args_1&block21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block21_layer_call_and_return_conditional_losses_431712!
block21/StatefulPartitionedCall?
block22/StatefulPartitionedCallStatefulPartitionedCall(block21/StatefulPartitionedCall:output:0&block22_statefulpartitionedcall_args_1&block22_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block22_layer_call_and_return_conditional_losses_431922!
block22/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(block22/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_432342,
*conv2d_transpose_1/StatefulPartitionedCall?
block10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0&block10_statefulpartitionedcall_args_1&block10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block10_layer_call_and_return_conditional_losses_432552!
block10/StatefulPartitionedCall?
IdentityIdentity(block10/StatefulPartitionedCall:output:0 ^block10/StatefulPartitionedCall ^block20/StatefulPartitionedCall ^block21/StatefulPartitionedCall ^block22/StatefulPartitionedCall ^block30/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2B
block10/StatefulPartitionedCallblock10/StatefulPartitionedCall2B
block20/StatefulPartitionedCallblock20/StatefulPartitionedCall2B
block21/StatefulPartitionedCallblock21/StatefulPartitionedCall2B
block22/StatefulPartitionedCallblock22/StatefulPartitionedCall2B
block30/StatefulPartitionedCallblock30/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
*__inference_sequential_layer_call_fn_43403
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
G
+__inference_block1_pool_layer_call_fn_42883

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_428772
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
ƀ
?#
__inference__traced_save_45096
file_prefix:
6savev2_block3__net_block_d1_kernel_read_readvariableop8
4savev2_block3__net_block_d1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_block3__net_sequential_block30_kernel_read_readvariableopB
>savev2_block3__net_sequential_block30_bias_read_readvariableopM
Isavev2_block3__net_sequential_conv2d_transpose_kernel_read_readvariableopK
Gsavev2_block3__net_sequential_conv2d_transpose_bias_read_readvariableopD
@savev2_block3__net_sequential_block20_kernel_read_readvariableopB
>savev2_block3__net_sequential_block20_bias_read_readvariableopD
@savev2_block3__net_sequential_block21_kernel_read_readvariableopB
>savev2_block3__net_sequential_block21_bias_read_readvariableopD
@savev2_block3__net_sequential_block22_kernel_read_readvariableopB
>savev2_block3__net_sequential_block22_bias_read_readvariableopO
Ksavev2_block3__net_sequential_conv2d_transpose_1_kernel_read_readvariableopM
Isavev2_block3__net_sequential_conv2d_transpose_1_bias_read_readvariableopD
@savev2_block3__net_sequential_block10_kernel_read_readvariableopB
>savev2_block3__net_sequential_block10_bias_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_block3__net_block_d1_kernel_m_read_readvariableop?
;savev2_adam_block3__net_block_d1_bias_m_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block30_kernel_m_read_readvariableopI
Esavev2_adam_block3__net_sequential_block30_bias_m_read_readvariableopT
Psavev2_adam_block3__net_sequential_conv2d_transpose_kernel_m_read_readvariableopR
Nsavev2_adam_block3__net_sequential_conv2d_transpose_bias_m_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block20_kernel_m_read_readvariableopI
Esavev2_adam_block3__net_sequential_block20_bias_m_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block21_kernel_m_read_readvariableopI
Esavev2_adam_block3__net_sequential_block21_bias_m_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block22_kernel_m_read_readvariableopI
Esavev2_adam_block3__net_sequential_block22_bias_m_read_readvariableopV
Rsavev2_adam_block3__net_sequential_conv2d_transpose_1_kernel_m_read_readvariableopT
Psavev2_adam_block3__net_sequential_conv2d_transpose_1_bias_m_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block10_kernel_m_read_readvariableopI
Esavev2_adam_block3__net_sequential_block10_bias_m_read_readvariableopA
=savev2_adam_block3__net_block_d1_kernel_v_read_readvariableop?
;savev2_adam_block3__net_block_d1_bias_v_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block30_kernel_v_read_readvariableopI
Esavev2_adam_block3__net_sequential_block30_bias_v_read_readvariableopT
Psavev2_adam_block3__net_sequential_conv2d_transpose_kernel_v_read_readvariableopR
Nsavev2_adam_block3__net_sequential_conv2d_transpose_bias_v_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block20_kernel_v_read_readvariableopI
Esavev2_adam_block3__net_sequential_block20_bias_v_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block21_kernel_v_read_readvariableopI
Esavev2_adam_block3__net_sequential_block21_bias_v_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block22_kernel_v_read_readvariableopI
Esavev2_adam_block3__net_sequential_block22_bias_v_read_readvariableopV
Rsavev2_adam_block3__net_sequential_conv2d_transpose_1_kernel_v_read_readvariableopT
Psavev2_adam_block3__net_sequential_conv2d_transpose_1_bias_v_read_readvariableopK
Gsavev2_adam_block3__net_sequential_block10_kernel_v_read_readvariableopI
Esavev2_adam_block3__net_sequential_block10_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1933909dd4254208b5a11ded9691a103/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

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
ShardedFilename? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB)conv_r1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv_r1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_block3__net_block_d1_kernel_read_readvariableop4savev2_block3__net_block_d1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_block3__net_sequential_block30_kernel_read_readvariableop>savev2_block3__net_sequential_block30_bias_read_readvariableopIsavev2_block3__net_sequential_conv2d_transpose_kernel_read_readvariableopGsavev2_block3__net_sequential_conv2d_transpose_bias_read_readvariableop@savev2_block3__net_sequential_block20_kernel_read_readvariableop>savev2_block3__net_sequential_block20_bias_read_readvariableop@savev2_block3__net_sequential_block21_kernel_read_readvariableop>savev2_block3__net_sequential_block21_bias_read_readvariableop@savev2_block3__net_sequential_block22_kernel_read_readvariableop>savev2_block3__net_sequential_block22_bias_read_readvariableopKsavev2_block3__net_sequential_conv2d_transpose_1_kernel_read_readvariableopIsavev2_block3__net_sequential_conv2d_transpose_1_bias_read_readvariableop@savev2_block3__net_sequential_block10_kernel_read_readvariableop>savev2_block3__net_sequential_block10_bias_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_block3__net_block_d1_kernel_m_read_readvariableop;savev2_adam_block3__net_block_d1_bias_m_read_readvariableopGsavev2_adam_block3__net_sequential_block30_kernel_m_read_readvariableopEsavev2_adam_block3__net_sequential_block30_bias_m_read_readvariableopPsavev2_adam_block3__net_sequential_conv2d_transpose_kernel_m_read_readvariableopNsavev2_adam_block3__net_sequential_conv2d_transpose_bias_m_read_readvariableopGsavev2_adam_block3__net_sequential_block20_kernel_m_read_readvariableopEsavev2_adam_block3__net_sequential_block20_bias_m_read_readvariableopGsavev2_adam_block3__net_sequential_block21_kernel_m_read_readvariableopEsavev2_adam_block3__net_sequential_block21_bias_m_read_readvariableopGsavev2_adam_block3__net_sequential_block22_kernel_m_read_readvariableopEsavev2_adam_block3__net_sequential_block22_bias_m_read_readvariableopRsavev2_adam_block3__net_sequential_conv2d_transpose_1_kernel_m_read_readvariableopPsavev2_adam_block3__net_sequential_conv2d_transpose_1_bias_m_read_readvariableopGsavev2_adam_block3__net_sequential_block10_kernel_m_read_readvariableopEsavev2_adam_block3__net_sequential_block10_bias_m_read_readvariableop=savev2_adam_block3__net_block_d1_kernel_v_read_readvariableop;savev2_adam_block3__net_block_d1_bias_v_read_readvariableopGsavev2_adam_block3__net_sequential_block30_kernel_v_read_readvariableopEsavev2_adam_block3__net_sequential_block30_bias_v_read_readvariableopPsavev2_adam_block3__net_sequential_conv2d_transpose_kernel_v_read_readvariableopNsavev2_adam_block3__net_sequential_conv2d_transpose_bias_v_read_readvariableopGsavev2_adam_block3__net_sequential_block20_kernel_v_read_readvariableopEsavev2_adam_block3__net_sequential_block20_bias_v_read_readvariableopGsavev2_adam_block3__net_sequential_block21_kernel_v_read_readvariableopEsavev2_adam_block3__net_sequential_block21_bias_v_read_readvariableopGsavev2_adam_block3__net_sequential_block22_kernel_v_read_readvariableopEsavev2_adam_block3__net_sequential_block22_bias_v_read_readvariableopRsavev2_adam_block3__net_sequential_conv2d_transpose_1_kernel_v_read_readvariableopPsavev2_adam_block3__net_sequential_conv2d_transpose_1_bias_v_read_readvariableopGsavev2_adam_block3__net_sequential_block10_kernel_v_read_readvariableopEsavev2_adam_block3__net_sequential_block10_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:: : : : : :??:?:??:?:??:?:??:?:??:?:@?:@:@@:@:@:@:@@:@:@?:?:??:?:??:?: : :@::??:?:??:?:??:?:??:?:??:?:@?:@:@@:@:@::??:?:??:?:??:?:??:?:??:?:@?:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
,__inference_block3_conv1_layer_call_fn_42958

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_429502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_42863

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?*
!__inference__traced_restore_45303
file_prefix0
,assignvariableop_block3__net_block_d1_kernel0
,assignvariableop_1_block3__net_block_d1_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate<
8assignvariableop_7_block3__net_sequential_block30_kernel:
6assignvariableop_8_block3__net_sequential_block30_biasE
Aassignvariableop_9_block3__net_sequential_conv2d_transpose_kernelD
@assignvariableop_10_block3__net_sequential_conv2d_transpose_bias=
9assignvariableop_11_block3__net_sequential_block20_kernel;
7assignvariableop_12_block3__net_sequential_block20_bias=
9assignvariableop_13_block3__net_sequential_block21_kernel;
7assignvariableop_14_block3__net_sequential_block21_bias=
9assignvariableop_15_block3__net_sequential_block22_kernel;
7assignvariableop_16_block3__net_sequential_block22_biasH
Dassignvariableop_17_block3__net_sequential_conv2d_transpose_1_kernelF
Bassignvariableop_18_block3__net_sequential_conv2d_transpose_1_bias=
9assignvariableop_19_block3__net_sequential_block10_kernel;
7assignvariableop_20_block3__net_sequential_block10_bias+
'assignvariableop_21_block1_conv1_kernel)
%assignvariableop_22_block1_conv1_bias+
'assignvariableop_23_block1_conv2_kernel)
%assignvariableop_24_block1_conv2_bias+
'assignvariableop_25_block2_conv1_kernel)
%assignvariableop_26_block2_conv1_bias+
'assignvariableop_27_block2_conv2_kernel)
%assignvariableop_28_block2_conv2_bias+
'assignvariableop_29_block3_conv1_kernel)
%assignvariableop_30_block3_conv1_bias
assignvariableop_31_total
assignvariableop_32_count:
6assignvariableop_33_adam_block3__net_block_d1_kernel_m8
4assignvariableop_34_adam_block3__net_block_d1_bias_mD
@assignvariableop_35_adam_block3__net_sequential_block30_kernel_mB
>assignvariableop_36_adam_block3__net_sequential_block30_bias_mM
Iassignvariableop_37_adam_block3__net_sequential_conv2d_transpose_kernel_mK
Gassignvariableop_38_adam_block3__net_sequential_conv2d_transpose_bias_mD
@assignvariableop_39_adam_block3__net_sequential_block20_kernel_mB
>assignvariableop_40_adam_block3__net_sequential_block20_bias_mD
@assignvariableop_41_adam_block3__net_sequential_block21_kernel_mB
>assignvariableop_42_adam_block3__net_sequential_block21_bias_mD
@assignvariableop_43_adam_block3__net_sequential_block22_kernel_mB
>assignvariableop_44_adam_block3__net_sequential_block22_bias_mO
Kassignvariableop_45_adam_block3__net_sequential_conv2d_transpose_1_kernel_mM
Iassignvariableop_46_adam_block3__net_sequential_conv2d_transpose_1_bias_mD
@assignvariableop_47_adam_block3__net_sequential_block10_kernel_mB
>assignvariableop_48_adam_block3__net_sequential_block10_bias_m:
6assignvariableop_49_adam_block3__net_block_d1_kernel_v8
4assignvariableop_50_adam_block3__net_block_d1_bias_vD
@assignvariableop_51_adam_block3__net_sequential_block30_kernel_vB
>assignvariableop_52_adam_block3__net_sequential_block30_bias_vM
Iassignvariableop_53_adam_block3__net_sequential_conv2d_transpose_kernel_vK
Gassignvariableop_54_adam_block3__net_sequential_conv2d_transpose_bias_vD
@assignvariableop_55_adam_block3__net_sequential_block20_kernel_vB
>assignvariableop_56_adam_block3__net_sequential_block20_bias_vD
@assignvariableop_57_adam_block3__net_sequential_block21_kernel_vB
>assignvariableop_58_adam_block3__net_sequential_block21_bias_vD
@assignvariableop_59_adam_block3__net_sequential_block22_kernel_vB
>assignvariableop_60_adam_block3__net_sequential_block22_bias_vO
Kassignvariableop_61_adam_block3__net_sequential_conv2d_transpose_1_kernel_vM
Iassignvariableop_62_adam_block3__net_sequential_conv2d_transpose_1_bias_vD
@assignvariableop_63_adam_block3__net_sequential_block10_kernel_vB
>assignvariableop_64_adam_block3__net_sequential_block10_bias_v
identity_66??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB)conv_r1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv_r1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEconv_r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv_r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_block3__net_block_d1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_block3__net_block_d1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_block3__net_sequential_block30_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_block3__net_sequential_block30_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpAassignvariableop_9_block3__net_sequential_conv2d_transpose_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp@assignvariableop_10_block3__net_sequential_conv2d_transpose_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_block3__net_sequential_block20_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_block3__net_sequential_block20_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_block3__net_sequential_block21_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_block3__net_sequential_block21_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp9assignvariableop_15_block3__net_sequential_block22_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_block3__net_sequential_block22_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpDassignvariableop_17_block3__net_sequential_conv2d_transpose_1_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpBassignvariableop_18_block3__net_sequential_conv2d_transpose_1_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_block3__net_sequential_block10_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_block3__net_sequential_block10_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block1_conv1_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block1_conv1_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block1_conv2_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block1_conv2_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block2_conv1_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block2_conv1_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block2_conv2_kernelIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block2_conv2_biasIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block3_conv1_kernelIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block3_conv1_biasIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_block3__net_block_d1_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_block3__net_block_d1_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp@assignvariableop_35_adam_block3__net_sequential_block30_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp>assignvariableop_36_adam_block3__net_sequential_block30_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpIassignvariableop_37_adam_block3__net_sequential_conv2d_transpose_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpGassignvariableop_38_adam_block3__net_sequential_conv2d_transpose_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_block3__net_sequential_block20_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_block3__net_sequential_block20_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp@assignvariableop_41_adam_block3__net_sequential_block21_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_block3__net_sequential_block21_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_block3__net_sequential_block22_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_block3__net_sequential_block22_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpKassignvariableop_45_adam_block3__net_sequential_conv2d_transpose_1_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpIassignvariableop_46_adam_block3__net_sequential_conv2d_transpose_1_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp@assignvariableop_47_adam_block3__net_sequential_block10_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_block3__net_sequential_block10_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_block3__net_block_d1_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_block3__net_block_d1_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp@assignvariableop_51_adam_block3__net_sequential_block30_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_block3__net_sequential_block30_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpIassignvariableop_53_adam_block3__net_sequential_conv2d_transpose_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpGassignvariableop_54_adam_block3__net_sequential_conv2d_transpose_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp@assignvariableop_55_adam_block3__net_sequential_block20_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp>assignvariableop_56_adam_block3__net_sequential_block20_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp@assignvariableop_57_adam_block3__net_sequential_block21_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp>assignvariableop_58_adam_block3__net_sequential_block21_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp@assignvariableop_59_adam_block3__net_sequential_block22_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_block3__net_sequential_block22_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpKassignvariableop_61_adam_block3__net_sequential_conv2d_transpose_1_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpIassignvariableop_62_adam_block3__net_sequential_conv2d_transpose_1_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp@assignvariableop_63_adam_block3__net_sequential_block10_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp>assignvariableop_64_adam_block3__net_sequential_block10_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_65?
Identity_66IdentityIdentity_65:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_66"#
identity_66Identity_66:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
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
AssignVariableOp_64AssignVariableOp_642(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_42931

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
B__inference_block10_layer_call_and_return_conditional_losses_43255

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?U
?
F__inference_block3__net_layer_call_and_return_conditional_losses_43785
input_tensor4
0original_vgg19_b3_statefulpartitionedcall_args_14
0original_vgg19_b3_statefulpartitionedcall_args_24
0original_vgg19_b3_statefulpartitionedcall_args_34
0original_vgg19_b3_statefulpartitionedcall_args_44
0original_vgg19_b3_statefulpartitionedcall_args_54
0original_vgg19_b3_statefulpartitionedcall_args_64
0original_vgg19_b3_statefulpartitionedcall_args_74
0original_vgg19_b3_statefulpartitionedcall_args_84
0original_vgg19_b3_statefulpartitionedcall_args_95
1original_vgg19_b3_statefulpartitionedcall_args_10-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10.
*sequential_statefulpartitionedcall_args_11.
*sequential_statefulpartitionedcall_args_12.
*sequential_statefulpartitionedcall_args_13.
*sequential_statefulpartitionedcall_args_14+
'block_d1_statefulpartitionedcall_args_1+
'block_d1_statefulpartitionedcall_args_2
identity?? Block_D1/StatefulPartitionedCall?)original_VGG19_B3/StatefulPartitionedCall?+original_VGG19_B3_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
)original_VGG19_B3/StatefulPartitionedCallStatefulPartitionedCallinput_tensor0original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_434722+
)original_VGG19_B3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall2original_VGG19_B3/StatefulPartitionedCall:output:0)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*sequential_statefulpartitionedcall_args_11*sequential_statefulpartitionedcall_args_12*sequential_statefulpartitionedcall_args_13*sequential_statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433422$
"sequential/StatefulPartitionedCall?
 Block_D1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0'block_d1_statefulpartitionedcall_args_1'block_d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_Block_D1_layer_call_and_return_conditional_losses_434152"
 Block_D1/StatefulPartitionedCall?
+original_VGG19_B3_1/StatefulPartitionedCallStatefulPartitionedCall)Block_D1/StatefulPartitionedCall:output:00original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*^original_VGG19_B3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430252-
+original_VGG19_B3_1/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference4original_VGG19_B3_1/StatefulPartitionedCall:output:02original_VGG19_B3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentity)Block_D1/StatefulPartitionedCall:output:0!^Block_D1/StatefulPartitionedCall*^original_VGG19_B3/StatefulPartitionedCall,^original_VGG19_B3_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2D
 Block_D1/StatefulPartitionedCall Block_D1/StatefulPartitionedCall2V
)original_VGG19_B3/StatefulPartitionedCall)original_VGG19_B3/StatefulPartitionedCall2Z
+original_VGG19_B3_1/StatefulPartitionedCall+original_VGG19_B3_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:, (
&
_user_specified_nameinput_tensor
?U
?
F__inference_block3__net_layer_call_and_return_conditional_losses_43670
input_14
0original_vgg19_b3_statefulpartitionedcall_args_14
0original_vgg19_b3_statefulpartitionedcall_args_24
0original_vgg19_b3_statefulpartitionedcall_args_34
0original_vgg19_b3_statefulpartitionedcall_args_44
0original_vgg19_b3_statefulpartitionedcall_args_54
0original_vgg19_b3_statefulpartitionedcall_args_64
0original_vgg19_b3_statefulpartitionedcall_args_74
0original_vgg19_b3_statefulpartitionedcall_args_84
0original_vgg19_b3_statefulpartitionedcall_args_95
1original_vgg19_b3_statefulpartitionedcall_args_10-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10.
*sequential_statefulpartitionedcall_args_11.
*sequential_statefulpartitionedcall_args_12.
*sequential_statefulpartitionedcall_args_13.
*sequential_statefulpartitionedcall_args_14+
'block_d1_statefulpartitionedcall_args_1+
'block_d1_statefulpartitionedcall_args_2
identity?? Block_D1/StatefulPartitionedCall?)original_VGG19_B3/StatefulPartitionedCall?+original_VGG19_B3_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
)original_VGG19_B3/StatefulPartitionedCallStatefulPartitionedCallinput_10original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_434722+
)original_VGG19_B3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall2original_VGG19_B3/StatefulPartitionedCall:output:0)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*sequential_statefulpartitionedcall_args_11*sequential_statefulpartitionedcall_args_12*sequential_statefulpartitionedcall_args_13*sequential_statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433422$
"sequential/StatefulPartitionedCall?
 Block_D1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0'block_d1_statefulpartitionedcall_args_1'block_d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_Block_D1_layer_call_and_return_conditional_losses_434152"
 Block_D1/StatefulPartitionedCall?
+original_VGG19_B3_1/StatefulPartitionedCallStatefulPartitionedCall)Block_D1/StatefulPartitionedCall:output:00original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*^original_VGG19_B3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430252-
+original_VGG19_B3_1/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference4original_VGG19_B3_1/StatefulPartitionedCall:output:02original_VGG19_B3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentity)Block_D1/StatefulPartitionedCall:output:0!^Block_D1/StatefulPartitionedCall*^original_VGG19_B3/StatefulPartitionedCall,^original_VGG19_B3_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2D
 Block_D1/StatefulPartitionedCall Block_D1/StatefulPartitionedCall2V
)original_VGG19_B3/StatefulPartitionedCall)original_VGG19_B3/StatefulPartitionedCall2Z
+original_VGG19_B3_1/StatefulPartitionedCall+original_VGG19_B3_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
??
?
F__inference_block3__net_layer_call_and_return_conditional_losses_44140
input_tensorA
=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resourceA
=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resourceB
>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resourceA
=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resourceA
=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resourceB
>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resourceA
=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource5
1sequential_block30_conv2d_readvariableop_resource6
2sequential_block30_biasadd_readvariableop_resourceH
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource?
;sequential_conv2d_transpose_biasadd_readvariableop_resource5
1sequential_block20_conv2d_readvariableop_resource6
2sequential_block20_biasadd_readvariableop_resource5
1sequential_block21_conv2d_readvariableop_resource6
2sequential_block21_biasadd_readvariableop_resource5
1sequential_block22_conv2d_readvariableop_resource6
2sequential_block22_biasadd_readvariableop_resourceJ
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceA
=sequential_conv2d_transpose_1_biasadd_readvariableop_resource5
1sequential_block10_conv2d_readvariableop_resource6
2sequential_block10_biasadd_readvariableop_resource+
'block_d1_conv2d_readvariableop_resource,
(block_d1_biasadd_readvariableop_resource
identity??Block_D1/BiasAdd/ReadVariableOp?Block_D1/Conv2D/ReadVariableOp?5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?)sequential/block10/BiasAdd/ReadVariableOp?(sequential/block10/Conv2D/ReadVariableOp?)sequential/block20/BiasAdd/ReadVariableOp?(sequential/block20/Conv2D/ReadVariableOp?)sequential/block21/BiasAdd/ReadVariableOp?(sequential/block21/Conv2D/ReadVariableOp?)sequential/block22/BiasAdd/ReadVariableOp?(sequential/block22/Conv2D/ReadVariableOp?)sequential/block30/BiasAdd/ReadVariableOp?(sequential/block30/Conv2D/ReadVariableOp?2sequential/conv2d_transpose/BiasAdd/ReadVariableOp?;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
,original_VGG19_B3/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block1_conv1/dilation_rate?
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype026
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block1_conv1/Conv2DConv2Dinput_tensor<original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%original_VGG19_B3/block1_conv1/Conv2D?
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block1_conv1/BiasAddBiasAdd.original_VGG19_B3/block1_conv1/Conv2D:output:0=original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&original_VGG19_B3/block1_conv1/BiasAdd?
#original_VGG19_B3/block1_conv1/ReluRelu/original_VGG19_B3/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#original_VGG19_B3/block1_conv1/Relu?
,original_VGG19_B3/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block1_conv2/dilation_rate?
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype026
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?
%original_VGG19_B3/block1_conv2/Conv2DConv2D1original_VGG19_B3/block1_conv1/Relu:activations:0<original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%original_VGG19_B3/block1_conv2/Conv2D?
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block1_conv2/BiasAddBiasAdd.original_VGG19_B3/block1_conv2/Conv2D:output:0=original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&original_VGG19_B3/block1_conv2/BiasAdd?
#original_VGG19_B3/block1_conv2/ReluRelu/original_VGG19_B3/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#original_VGG19_B3/block1_conv2/Relu?
%original_VGG19_B3/block1_pool/MaxPoolMaxPool1original_VGG19_B3/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2'
%original_VGG19_B3/block1_pool/MaxPool?
,original_VGG19_B3/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block2_conv1/dilation_rate?
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype026
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block2_conv1/Conv2DConv2D.original_VGG19_B3/block1_pool/MaxPool:output:0<original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%original_VGG19_B3/block2_conv1/Conv2D?
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block2_conv1/BiasAddBiasAdd.original_VGG19_B3/block2_conv1/Conv2D:output:0=original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&original_VGG19_B3/block2_conv1/BiasAdd?
#original_VGG19_B3/block2_conv1/ReluRelu/original_VGG19_B3/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#original_VGG19_B3/block2_conv1/Relu?
,original_VGG19_B3/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block2_conv2/dilation_rate?
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?
%original_VGG19_B3/block2_conv2/Conv2DConv2D1original_VGG19_B3/block2_conv1/Relu:activations:0<original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%original_VGG19_B3/block2_conv2/Conv2D?
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block2_conv2/BiasAddBiasAdd.original_VGG19_B3/block2_conv2/Conv2D:output:0=original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&original_VGG19_B3/block2_conv2/BiasAdd?
#original_VGG19_B3/block2_conv2/ReluRelu/original_VGG19_B3/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#original_VGG19_B3/block2_conv2/Relu?
%original_VGG19_B3/block2_pool/MaxPoolMaxPool1original_VGG19_B3/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2'
%original_VGG19_B3/block2_pool/MaxPool?
,original_VGG19_B3/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block3_conv1/dilation_rate?
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block3_conv1/Conv2DConv2D.original_VGG19_B3/block2_pool/MaxPool:output:0<original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2'
%original_VGG19_B3/block3_conv1/Conv2D?
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block3_conv1/BiasAddBiasAdd.original_VGG19_B3/block3_conv1/Conv2D:output:0=original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2(
&original_VGG19_B3/block3_conv1/BiasAdd?
#original_VGG19_B3/block3_conv1/ReluRelu/original_VGG19_B3/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2%
#original_VGG19_B3/block3_conv1/Relu?
(sequential/block30/Conv2D/ReadVariableOpReadVariableOp1sequential_block30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block30/Conv2D/ReadVariableOp?
sequential/block30/Conv2DConv2D1original_VGG19_B3/block3_conv1/Relu:activations:00sequential/block30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
sequential/block30/Conv2D?
)sequential/block30/BiasAdd/ReadVariableOpReadVariableOp2sequential_block30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block30/BiasAdd/ReadVariableOp?
sequential/block30/BiasAddBiasAdd"sequential/block30/Conv2D:output:01sequential/block30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
sequential/block30/BiasAdd?
sequential/block30/ReluRelu#sequential/block30/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
sequential/block30/Relu?
!sequential/conv2d_transpose/ShapeShape%sequential/block30/Relu:activations:0*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/Shape?
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/conv2d_transpose/strided_slice/stack?
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_1?
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_2?
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential/conv2d_transpose/strided_slice?
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice_1/stack?
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_1?
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_2?
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/Shape:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_1?
1sequential/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice_2/stack?
3sequential/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_2/stack_1?
3sequential/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_2/stack_2?
+sequential/conv2d_transpose/strided_slice_2StridedSlice*sequential/conv2d_transpose/Shape:output:0:sequential/conv2d_transpose/strided_slice_2/stack:output:0<sequential/conv2d_transpose/strided_slice_2/stack_1:output:0<sequential/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_2?
!sequential/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/conv2d_transpose/mul/y?
sequential/conv2d_transpose/mulMul4sequential/conv2d_transpose/strided_slice_1:output:0*sequential/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/conv2d_transpose/mul?
#sequential/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/mul_1/y?
!sequential/conv2d_transpose/mul_1Mul4sequential/conv2d_transpose/strided_slice_2:output:0,sequential/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/conv2d_transpose/mul_1?
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential/conv2d_transpose/stack/3?
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0#sequential/conv2d_transpose/mul:z:0%sequential/conv2d_transpose/mul_1:z:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/stack?
1sequential/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose/strided_slice_3/stack?
3sequential/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_3/stack_1?
3sequential/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_3/stack_2?
+sequential/conv2d_transpose/strided_slice_3StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_3/stack:output:0<sequential/conv2d_transpose/strided_slice_3/stack_1:output:0<sequential/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_3?
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02=
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential/block30/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2.
,sequential/conv2d_transpose/conv2d_transpose?
2sequential/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;sequential_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp?
#sequential/conv2d_transpose/BiasAddBiasAdd5sequential/conv2d_transpose/conv2d_transpose:output:0:sequential/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2%
#sequential/conv2d_transpose/BiasAdd?
(sequential/block20/Conv2D/ReadVariableOpReadVariableOp1sequential_block20_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block20/Conv2D/ReadVariableOp?
sequential/block20/Conv2DConv2D,sequential/conv2d_transpose/BiasAdd:output:00sequential/block20/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block20/Conv2D?
)sequential/block20/BiasAdd/ReadVariableOpReadVariableOp2sequential_block20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block20/BiasAdd/ReadVariableOp?
sequential/block20/BiasAddBiasAdd"sequential/block20/Conv2D:output:01sequential/block20/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block20/BiasAdd?
sequential/block20/ReluRelu#sequential/block20/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block20/Relu?
(sequential/block21/Conv2D/ReadVariableOpReadVariableOp1sequential_block21_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block21/Conv2D/ReadVariableOp?
sequential/block21/Conv2DConv2D%sequential/block20/Relu:activations:00sequential/block21/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block21/Conv2D?
)sequential/block21/BiasAdd/ReadVariableOpReadVariableOp2sequential_block21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block21/BiasAdd/ReadVariableOp?
sequential/block21/BiasAddBiasAdd"sequential/block21/Conv2D:output:01sequential/block21/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block21/BiasAdd?
sequential/block21/ReluRelu#sequential/block21/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block21/Relu?
(sequential/block22/Conv2D/ReadVariableOpReadVariableOp1sequential_block22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block22/Conv2D/ReadVariableOp?
sequential/block22/Conv2DConv2D%sequential/block21/Relu:activations:00sequential/block22/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block22/Conv2D?
)sequential/block22/BiasAdd/ReadVariableOpReadVariableOp2sequential_block22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block22/BiasAdd/ReadVariableOp?
sequential/block22/BiasAddBiasAdd"sequential/block22/Conv2D:output:01sequential/block22/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block22/BiasAdd?
sequential/block22/ReluRelu#sequential/block22/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block22/Relu?
#sequential/conv2d_transpose_1/ShapeShape%sequential/block22/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/Shape?
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_1/strided_slice/stack?
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_1?
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_2?
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_1/strided_slice?
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice_1/stack?
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_1?
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_2?
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/Shape:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_1?
3sequential/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice_2/stack?
5sequential/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_2/stack_1?
5sequential/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_2/stack_2?
-sequential/conv2d_transpose_1/strided_slice_2StridedSlice,sequential/conv2d_transpose_1/Shape:output:0<sequential/conv2d_transpose_1/strided_slice_2/stack:output:0>sequential/conv2d_transpose_1/strided_slice_2/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_2?
#sequential/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose_1/mul/y?
!sequential/conv2d_transpose_1/mulMul6sequential/conv2d_transpose_1/strided_slice_1:output:0,sequential/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/conv2d_transpose_1/mul?
%sequential/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/mul_1/y?
#sequential/conv2d_transpose_1/mul_1Mul6sequential/conv2d_transpose_1/strided_slice_2:output:0.sequential/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential/conv2d_transpose_1/mul_1?
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential/conv2d_transpose_1/stack/3?
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0%sequential/conv2d_transpose_1/mul:z:0'sequential/conv2d_transpose_1/mul_1:z:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/stack?
3sequential/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_1/strided_slice_3/stack?
5sequential/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_3/stack_1?
5sequential/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_3/stack_2?
-sequential/conv2d_transpose_1/strided_slice_3StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_3/stack:output:0>sequential/conv2d_transpose_1/strided_slice_3/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_3?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0%sequential/block22/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
20
.sequential/conv2d_transpose_1/conv2d_transpose?
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?
%sequential/conv2d_transpose_1/BiasAddBiasAdd7sequential/conv2d_transpose_1/conv2d_transpose:output:0<sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2'
%sequential/conv2d_transpose_1/BiasAdd?
(sequential/block10/Conv2D/ReadVariableOpReadVariableOp1sequential_block10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(sequential/block10/Conv2D/ReadVariableOp?
sequential/block10/Conv2DConv2D.sequential/conv2d_transpose_1/BiasAdd:output:00sequential/block10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
sequential/block10/Conv2D?
)sequential/block10/BiasAdd/ReadVariableOpReadVariableOp2sequential_block10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/block10/BiasAdd/ReadVariableOp?
sequential/block10/BiasAddBiasAdd"sequential/block10/Conv2D:output:01sequential/block10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
sequential/block10/BiasAdd?
sequential/block10/ReluRelu#sequential/block10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
sequential/block10/Relu?
Block_D1/Conv2D/ReadVariableOpReadVariableOp'block_d1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
Block_D1/Conv2D/ReadVariableOp?
Block_D1/Conv2DConv2D%sequential/block10/Relu:activations:0&Block_D1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Block_D1/Conv2D?
Block_D1/BiasAdd/ReadVariableOpReadVariableOp(block_d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Block_D1/BiasAdd/ReadVariableOp?
Block_D1/BiasAddBiasAddBlock_D1/Conv2D:output:0'Block_D1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Block_D1/BiasAdd?
.original_VGG19_B3_1/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block1_conv1/dilation_rate?
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype028
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block1_conv1/Conv2DConv2DBlock_D1/BiasAdd:output:0>original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block1_conv1/Conv2D?
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype029
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block1_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block1_conv1/Conv2D:output:0?original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2*
(original_VGG19_B3_1/block1_conv1/BiasAdd?
%original_VGG19_B3_1/block1_conv1/ReluRelu1original_VGG19_B3_1/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2'
%original_VGG19_B3_1/block1_conv1/Relu?
.original_VGG19_B3_1/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block1_conv2/dilation_rate?
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resource5^original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype028
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block1_conv2/Conv2DConv2D3original_VGG19_B3_1/block1_conv1/Relu:activations:0>original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block1_conv2/Conv2D?
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resource6^original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype029
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block1_conv2/BiasAddBiasAdd0original_VGG19_B3_1/block1_conv2/Conv2D:output:0?original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2*
(original_VGG19_B3_1/block1_conv2/BiasAdd?
%original_VGG19_B3_1/block1_conv2/ReluRelu1original_VGG19_B3_1/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2'
%original_VGG19_B3_1/block1_conv2/Relu?
'original_VGG19_B3_1/block1_pool/MaxPoolMaxPool3original_VGG19_B3_1/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2)
'original_VGG19_B3_1/block1_pool/MaxPool?
.original_VGG19_B3_1/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block2_conv1/dilation_rate?
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp*'
_output_shapes
:@?*
dtype028
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block2_conv1/Conv2DConv2D0original_VGG19_B3_1/block1_pool/MaxPool:output:0>original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block2_conv1/Conv2D?
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block2_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block2_conv1/Conv2D:output:0?original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2*
(original_VGG19_B3_1/block2_conv1/BiasAdd?
%original_VGG19_B3_1/block2_conv1/ReluRelu1original_VGG19_B3_1/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2'
%original_VGG19_B3_1/block2_conv1/Relu?
.original_VGG19_B3_1/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block2_conv2/dilation_rate?
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resource5^original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype028
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block2_conv2/Conv2DConv2D3original_VGG19_B3_1/block2_conv1/Relu:activations:0>original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block2_conv2/Conv2D?
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resource6^original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block2_conv2/BiasAddBiasAdd0original_VGG19_B3_1/block2_conv2/Conv2D:output:0?original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2*
(original_VGG19_B3_1/block2_conv2/BiasAdd?
%original_VGG19_B3_1/block2_conv2/ReluRelu1original_VGG19_B3_1/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2'
%original_VGG19_B3_1/block2_conv2/Relu?
'original_VGG19_B3_1/block2_pool/MaxPoolMaxPool3original_VGG19_B3_1/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2)
'original_VGG19_B3_1/block2_pool/MaxPool?
.original_VGG19_B3_1/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block3_conv1/dilation_rate?
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype028
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block3_conv1/Conv2DConv2D0original_VGG19_B3_1/block2_pool/MaxPool:output:0>original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block3_conv1/Conv2D?
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block3_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block3_conv1/Conv2D:output:0?original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2*
(original_VGG19_B3_1/block3_conv1/BiasAdd?
%original_VGG19_B3_1/block3_conv1/ReluRelu1original_VGG19_B3_1/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2'
%original_VGG19_B3_1/block3_conv1/Relu?
$mean_squared_error/SquaredDifferenceSquaredDifference3original_VGG19_B3_1/block3_conv1/Relu:activations:01original_VGG19_B3/block3_conv1/Relu:activations:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentityBlock_D1/BiasAdd:output:0 ^Block_D1/BiasAdd/ReadVariableOp^Block_D1/Conv2D/ReadVariableOp6^original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp6^original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp5^original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp6^original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp6^original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp5^original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp6^original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp*^sequential/block10/BiasAdd/ReadVariableOp)^sequential/block10/Conv2D/ReadVariableOp*^sequential/block20/BiasAdd/ReadVariableOp)^sequential/block20/Conv2D/ReadVariableOp*^sequential/block21/BiasAdd/ReadVariableOp)^sequential/block21/Conv2D/ReadVariableOp*^sequential/block22/BiasAdd/ReadVariableOp)^sequential/block22/Conv2D/ReadVariableOp*^sequential/block30/BiasAdd/ReadVariableOp)^sequential/block30/Conv2D/ReadVariableOp3^sequential/conv2d_transpose/BiasAdd/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2B
Block_D1/BiasAdd/ReadVariableOpBlock_D1/BiasAdd/ReadVariableOp2@
Block_D1/Conv2D/ReadVariableOpBlock_D1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp2V
)sequential/block10/BiasAdd/ReadVariableOp)sequential/block10/BiasAdd/ReadVariableOp2T
(sequential/block10/Conv2D/ReadVariableOp(sequential/block10/Conv2D/ReadVariableOp2V
)sequential/block20/BiasAdd/ReadVariableOp)sequential/block20/BiasAdd/ReadVariableOp2T
(sequential/block20/Conv2D/ReadVariableOp(sequential/block20/Conv2D/ReadVariableOp2V
)sequential/block21/BiasAdd/ReadVariableOp)sequential/block21/BiasAdd/ReadVariableOp2T
(sequential/block21/Conv2D/ReadVariableOp(sequential/block21/Conv2D/ReadVariableOp2V
)sequential/block22/BiasAdd/ReadVariableOp)sequential/block22/BiasAdd/ReadVariableOp2T
(sequential/block22/Conv2D/ReadVariableOp(sequential/block22/Conv2D/ReadVariableOp2V
)sequential/block30/BiasAdd/ReadVariableOp)sequential/block30/BiasAdd/ReadVariableOp2T
(sequential/block30/Conv2D/ReadVariableOp(sequential/block30/Conv2D/ReadVariableOp2h
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:, (
&
_user_specified_nameinput_tensor
?
?
'__inference_block20_layer_call_fn_43158

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block20_layer_call_and_return_conditional_losses_431502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?

?
C__inference_Block_D1_layer_call_and_return_conditional_losses_43415

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?B
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44447

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*B
_output_shapes0
.:,????????????????????????????*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_44858

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?U
?
F__inference_block3__net_layer_call_and_return_conditional_losses_43726
input_14
0original_vgg19_b3_statefulpartitionedcall_args_14
0original_vgg19_b3_statefulpartitionedcall_args_24
0original_vgg19_b3_statefulpartitionedcall_args_34
0original_vgg19_b3_statefulpartitionedcall_args_44
0original_vgg19_b3_statefulpartitionedcall_args_54
0original_vgg19_b3_statefulpartitionedcall_args_64
0original_vgg19_b3_statefulpartitionedcall_args_74
0original_vgg19_b3_statefulpartitionedcall_args_84
0original_vgg19_b3_statefulpartitionedcall_args_95
1original_vgg19_b3_statefulpartitionedcall_args_10-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10.
*sequential_statefulpartitionedcall_args_11.
*sequential_statefulpartitionedcall_args_12.
*sequential_statefulpartitionedcall_args_13.
*sequential_statefulpartitionedcall_args_14+
'block_d1_statefulpartitionedcall_args_1+
'block_d1_statefulpartitionedcall_args_2
identity?? Block_D1/StatefulPartitionedCall?)original_VGG19_B3/StatefulPartitionedCall?+original_VGG19_B3_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
)original_VGG19_B3/StatefulPartitionedCallStatefulPartitionedCallinput_10original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_435182+
)original_VGG19_B3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall2original_VGG19_B3/StatefulPartitionedCall:output:0)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*sequential_statefulpartitionedcall_args_11*sequential_statefulpartitionedcall_args_12*sequential_statefulpartitionedcall_args_13*sequential_statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433862$
"sequential/StatefulPartitionedCall?
 Block_D1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0'block_d1_statefulpartitionedcall_args_1'block_d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_Block_D1_layer_call_and_return_conditional_losses_434152"
 Block_D1/StatefulPartitionedCall?
+original_VGG19_B3_1/StatefulPartitionedCallStatefulPartitionedCall)Block_D1/StatefulPartitionedCall:output:00original_vgg19_b3_statefulpartitionedcall_args_10original_vgg19_b3_statefulpartitionedcall_args_20original_vgg19_b3_statefulpartitionedcall_args_30original_vgg19_b3_statefulpartitionedcall_args_40original_vgg19_b3_statefulpartitionedcall_args_50original_vgg19_b3_statefulpartitionedcall_args_60original_vgg19_b3_statefulpartitionedcall_args_70original_vgg19_b3_statefulpartitionedcall_args_80original_vgg19_b3_statefulpartitionedcall_args_91original_vgg19_b3_statefulpartitionedcall_args_10*^original_VGG19_B3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430612-
+original_VGG19_B3_1/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference4original_VGG19_B3_1/StatefulPartitionedCall:output:02original_VGG19_B3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentity)Block_D1/StatefulPartitionedCall:output:0!^Block_D1/StatefulPartitionedCall*^original_VGG19_B3/StatefulPartitionedCall,^original_VGG19_B3_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2D
 Block_D1/StatefulPartitionedCall Block_D1/StatefulPartitionedCall2V
)original_VGG19_B3/StatefulPartitionedCall)original_VGG19_B3/StatefulPartitionedCall2Z
+original_VGG19_B3_1/StatefulPartitionedCall+original_VGG19_B3_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
'__inference_block21_layer_call_fn_43179

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block21_layer_call_and_return_conditional_losses_431712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_Block_D1_layer_call_fn_43423

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_Block_D1_layer_call_and_return_conditional_losses_434152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
F__inference_block3__net_layer_call_and_return_conditional_losses_44339
input_tensorA
=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resourceA
=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resourceB
>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resourceA
=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resourceA
=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resourceB
>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resourceA
=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resourceB
>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource5
1sequential_block30_conv2d_readvariableop_resource6
2sequential_block30_biasadd_readvariableop_resourceH
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource?
;sequential_conv2d_transpose_biasadd_readvariableop_resource5
1sequential_block20_conv2d_readvariableop_resource6
2sequential_block20_biasadd_readvariableop_resource5
1sequential_block21_conv2d_readvariableop_resource6
2sequential_block21_biasadd_readvariableop_resource5
1sequential_block22_conv2d_readvariableop_resource6
2sequential_block22_biasadd_readvariableop_resourceJ
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceA
=sequential_conv2d_transpose_1_biasadd_readvariableop_resource5
1sequential_block10_conv2d_readvariableop_resource6
2sequential_block10_biasadd_readvariableop_resource+
'block_d1_conv2d_readvariableop_resource,
(block_d1_biasadd_readvariableop_resource
identity??Block_D1/BiasAdd/ReadVariableOp?Block_D1/Conv2D/ReadVariableOp?5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?)sequential/block10/BiasAdd/ReadVariableOp?(sequential/block10/Conv2D/ReadVariableOp?)sequential/block20/BiasAdd/ReadVariableOp?(sequential/block20/Conv2D/ReadVariableOp?)sequential/block21/BiasAdd/ReadVariableOp?(sequential/block21/Conv2D/ReadVariableOp?)sequential/block22/BiasAdd/ReadVariableOp?(sequential/block22/Conv2D/ReadVariableOp?)sequential/block30/BiasAdd/ReadVariableOp?(sequential/block30/Conv2D/ReadVariableOp?2sequential/conv2d_transpose/BiasAdd/ReadVariableOp?;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
,original_VGG19_B3/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block1_conv1/dilation_rate?
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype026
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block1_conv1/Conv2DConv2Dinput_tensor<original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%original_VGG19_B3/block1_conv1/Conv2D?
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block1_conv1/BiasAddBiasAdd.original_VGG19_B3/block1_conv1/Conv2D:output:0=original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&original_VGG19_B3/block1_conv1/BiasAdd?
#original_VGG19_B3/block1_conv1/ReluRelu/original_VGG19_B3/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#original_VGG19_B3/block1_conv1/Relu?
,original_VGG19_B3/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block1_conv2/dilation_rate?
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype026
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp?
%original_VGG19_B3/block1_conv2/Conv2DConv2D1original_VGG19_B3/block1_conv1/Relu:activations:0<original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%original_VGG19_B3/block1_conv2/Conv2D?
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block1_conv2/BiasAddBiasAdd.original_VGG19_B3/block1_conv2/Conv2D:output:0=original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&original_VGG19_B3/block1_conv2/BiasAdd?
#original_VGG19_B3/block1_conv2/ReluRelu/original_VGG19_B3/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2%
#original_VGG19_B3/block1_conv2/Relu?
%original_VGG19_B3/block1_pool/MaxPoolMaxPool1original_VGG19_B3/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2'
%original_VGG19_B3/block1_pool/MaxPool?
,original_VGG19_B3/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block2_conv1/dilation_rate?
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype026
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block2_conv1/Conv2DConv2D.original_VGG19_B3/block1_pool/MaxPool:output:0<original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%original_VGG19_B3/block2_conv1/Conv2D?
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block2_conv1/BiasAddBiasAdd.original_VGG19_B3/block2_conv1/Conv2D:output:0=original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&original_VGG19_B3/block2_conv1/BiasAdd?
#original_VGG19_B3/block2_conv1/ReluRelu/original_VGG19_B3/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#original_VGG19_B3/block2_conv1/Relu?
,original_VGG19_B3/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block2_conv2/dilation_rate?
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp?
%original_VGG19_B3/block2_conv2/Conv2DConv2D1original_VGG19_B3/block2_conv1/Relu:activations:0<original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2'
%original_VGG19_B3/block2_conv2/Conv2D?
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block2_conv2/BiasAddBiasAdd.original_VGG19_B3/block2_conv2/Conv2D:output:0=original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2(
&original_VGG19_B3/block2_conv2/BiasAdd?
#original_VGG19_B3/block2_conv2/ReluRelu/original_VGG19_B3/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2%
#original_VGG19_B3/block2_conv2/Relu?
%original_VGG19_B3/block2_pool/MaxPoolMaxPool1original_VGG19_B3/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2'
%original_VGG19_B3/block2_pool/MaxPool?
,original_VGG19_B3/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,original_VGG19_B3/block3_conv1/dilation_rate?
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype026
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp?
%original_VGG19_B3/block3_conv1/Conv2DConv2D.original_VGG19_B3/block2_pool/MaxPool:output:0<original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2'
%original_VGG19_B3/block3_conv1/Conv2D?
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp?
&original_VGG19_B3/block3_conv1/BiasAddBiasAdd.original_VGG19_B3/block3_conv1/Conv2D:output:0=original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2(
&original_VGG19_B3/block3_conv1/BiasAdd?
#original_VGG19_B3/block3_conv1/ReluRelu/original_VGG19_B3/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2%
#original_VGG19_B3/block3_conv1/Relu?
(sequential/block30/Conv2D/ReadVariableOpReadVariableOp1sequential_block30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block30/Conv2D/ReadVariableOp?
sequential/block30/Conv2DConv2D1original_VGG19_B3/block3_conv1/Relu:activations:00sequential/block30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
sequential/block30/Conv2D?
)sequential/block30/BiasAdd/ReadVariableOpReadVariableOp2sequential_block30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block30/BiasAdd/ReadVariableOp?
sequential/block30/BiasAddBiasAdd"sequential/block30/Conv2D:output:01sequential/block30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
sequential/block30/BiasAdd?
sequential/block30/ReluRelu#sequential/block30/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
sequential/block30/Relu?
!sequential/conv2d_transpose/ShapeShape%sequential/block30/Relu:activations:0*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/Shape?
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/conv2d_transpose/strided_slice/stack?
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_1?
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_2?
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential/conv2d_transpose/strided_slice?
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice_1/stack?
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_1?
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_2?
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/Shape:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_1?
1sequential/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice_2/stack?
3sequential/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_2/stack_1?
3sequential/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_2/stack_2?
+sequential/conv2d_transpose/strided_slice_2StridedSlice*sequential/conv2d_transpose/Shape:output:0:sequential/conv2d_transpose/strided_slice_2/stack:output:0<sequential/conv2d_transpose/strided_slice_2/stack_1:output:0<sequential/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_2?
!sequential/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/conv2d_transpose/mul/y?
sequential/conv2d_transpose/mulMul4sequential/conv2d_transpose/strided_slice_1:output:0*sequential/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/conv2d_transpose/mul?
#sequential/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/mul_1/y?
!sequential/conv2d_transpose/mul_1Mul4sequential/conv2d_transpose/strided_slice_2:output:0,sequential/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/conv2d_transpose/mul_1?
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential/conv2d_transpose/stack/3?
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0#sequential/conv2d_transpose/mul:z:0%sequential/conv2d_transpose/mul_1:z:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/stack?
1sequential/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose/strided_slice_3/stack?
3sequential/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_3/stack_1?
3sequential/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_3/stack_2?
+sequential/conv2d_transpose/strided_slice_3StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_3/stack:output:0<sequential/conv2d_transpose/strided_slice_3/stack_1:output:0<sequential/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_3?
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02=
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential/block30/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2.
,sequential/conv2d_transpose/conv2d_transpose?
2sequential/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;sequential_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp?
#sequential/conv2d_transpose/BiasAddBiasAdd5sequential/conv2d_transpose/conv2d_transpose:output:0:sequential/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2%
#sequential/conv2d_transpose/BiasAdd?
(sequential/block20/Conv2D/ReadVariableOpReadVariableOp1sequential_block20_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block20/Conv2D/ReadVariableOp?
sequential/block20/Conv2DConv2D,sequential/conv2d_transpose/BiasAdd:output:00sequential/block20/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block20/Conv2D?
)sequential/block20/BiasAdd/ReadVariableOpReadVariableOp2sequential_block20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block20/BiasAdd/ReadVariableOp?
sequential/block20/BiasAddBiasAdd"sequential/block20/Conv2D:output:01sequential/block20/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block20/BiasAdd?
sequential/block20/ReluRelu#sequential/block20/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block20/Relu?
(sequential/block21/Conv2D/ReadVariableOpReadVariableOp1sequential_block21_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block21/Conv2D/ReadVariableOp?
sequential/block21/Conv2DConv2D%sequential/block20/Relu:activations:00sequential/block21/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block21/Conv2D?
)sequential/block21/BiasAdd/ReadVariableOpReadVariableOp2sequential_block21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block21/BiasAdd/ReadVariableOp?
sequential/block21/BiasAddBiasAdd"sequential/block21/Conv2D:output:01sequential/block21/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block21/BiasAdd?
sequential/block21/ReluRelu#sequential/block21/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block21/Relu?
(sequential/block22/Conv2D/ReadVariableOpReadVariableOp1sequential_block22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/block22/Conv2D/ReadVariableOp?
sequential/block22/Conv2DConv2D%sequential/block21/Relu:activations:00sequential/block22/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential/block22/Conv2D?
)sequential/block22/BiasAdd/ReadVariableOpReadVariableOp2sequential_block22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/block22/BiasAdd/ReadVariableOp?
sequential/block22/BiasAddBiasAdd"sequential/block22/Conv2D:output:01sequential/block22/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
sequential/block22/BiasAdd?
sequential/block22/ReluRelu#sequential/block22/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential/block22/Relu?
#sequential/conv2d_transpose_1/ShapeShape%sequential/block22/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/Shape?
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_1/strided_slice/stack?
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_1?
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_2?
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_1/strided_slice?
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice_1/stack?
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_1?
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_2?
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/Shape:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_1?
3sequential/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice_2/stack?
5sequential/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_2/stack_1?
5sequential/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_2/stack_2?
-sequential/conv2d_transpose_1/strided_slice_2StridedSlice,sequential/conv2d_transpose_1/Shape:output:0<sequential/conv2d_transpose_1/strided_slice_2/stack:output:0>sequential/conv2d_transpose_1/strided_slice_2/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_2?
#sequential/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose_1/mul/y?
!sequential/conv2d_transpose_1/mulMul6sequential/conv2d_transpose_1/strided_slice_1:output:0,sequential/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/conv2d_transpose_1/mul?
%sequential/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/mul_1/y?
#sequential/conv2d_transpose_1/mul_1Mul6sequential/conv2d_transpose_1/strided_slice_2:output:0.sequential/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential/conv2d_transpose_1/mul_1?
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential/conv2d_transpose_1/stack/3?
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0%sequential/conv2d_transpose_1/mul:z:0'sequential/conv2d_transpose_1/mul_1:z:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/stack?
3sequential/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_1/strided_slice_3/stack?
5sequential/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_3/stack_1?
5sequential/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_3/stack_2?
-sequential/conv2d_transpose_1/strided_slice_3StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_3/stack:output:0>sequential/conv2d_transpose_1/strided_slice_3/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_3?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0%sequential/block22/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
20
.sequential/conv2d_transpose_1/conv2d_transpose?
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?
%sequential/conv2d_transpose_1/BiasAddBiasAdd7sequential/conv2d_transpose_1/conv2d_transpose:output:0<sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2'
%sequential/conv2d_transpose_1/BiasAdd?
(sequential/block10/Conv2D/ReadVariableOpReadVariableOp1sequential_block10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(sequential/block10/Conv2D/ReadVariableOp?
sequential/block10/Conv2DConv2D.sequential/conv2d_transpose_1/BiasAdd:output:00sequential/block10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
sequential/block10/Conv2D?
)sequential/block10/BiasAdd/ReadVariableOpReadVariableOp2sequential_block10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/block10/BiasAdd/ReadVariableOp?
sequential/block10/BiasAddBiasAdd"sequential/block10/Conv2D:output:01sequential/block10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
sequential/block10/BiasAdd?
sequential/block10/ReluRelu#sequential/block10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
sequential/block10/Relu?
Block_D1/Conv2D/ReadVariableOpReadVariableOp'block_d1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
Block_D1/Conv2D/ReadVariableOp?
Block_D1/Conv2DConv2D%sequential/block10/Relu:activations:0&Block_D1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Block_D1/Conv2D?
Block_D1/BiasAdd/ReadVariableOpReadVariableOp(block_d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Block_D1/BiasAdd/ReadVariableOp?
Block_D1/BiasAddBiasAddBlock_D1/Conv2D:output:0'Block_D1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Block_D1/BiasAdd?
.original_VGG19_B3_1/block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block1_conv1/dilation_rate?
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype028
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block1_conv1/Conv2DConv2DBlock_D1/BiasAdd:output:0>original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block1_conv1/Conv2D?
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype029
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block1_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block1_conv1/Conv2D:output:0?original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2*
(original_VGG19_B3_1/block1_conv1/BiasAdd?
%original_VGG19_B3_1/block1_conv1/ReluRelu1original_VGG19_B3_1/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2'
%original_VGG19_B3_1/block1_conv1/Relu?
.original_VGG19_B3_1/block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block1_conv2/dilation_rate?
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block1_conv2_conv2d_readvariableop_resource5^original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype028
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block1_conv2/Conv2DConv2D3original_VGG19_B3_1/block1_conv1/Relu:activations:0>original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block1_conv2/Conv2D?
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block1_conv2_biasadd_readvariableop_resource6^original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp*
_output_shapes
:@*
dtype029
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block1_conv2/BiasAddBiasAdd0original_VGG19_B3_1/block1_conv2/Conv2D:output:0?original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2*
(original_VGG19_B3_1/block1_conv2/BiasAdd?
%original_VGG19_B3_1/block1_conv2/ReluRelu1original_VGG19_B3_1/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2'
%original_VGG19_B3_1/block1_conv2/Relu?
'original_VGG19_B3_1/block1_pool/MaxPoolMaxPool3original_VGG19_B3_1/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2)
'original_VGG19_B3_1/block1_pool/MaxPool?
.original_VGG19_B3_1/block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block2_conv1/dilation_rate?
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp*'
_output_shapes
:@?*
dtype028
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block2_conv1/Conv2DConv2D0original_VGG19_B3_1/block1_pool/MaxPool:output:0>original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block2_conv1/Conv2D?
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block2_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block2_conv1/Conv2D:output:0?original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2*
(original_VGG19_B3_1/block2_conv1/BiasAdd?
%original_VGG19_B3_1/block2_conv1/ReluRelu1original_VGG19_B3_1/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2'
%original_VGG19_B3_1/block2_conv1/Relu?
.original_VGG19_B3_1/block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block2_conv2/dilation_rate?
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block2_conv2_conv2d_readvariableop_resource5^original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype028
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block2_conv2/Conv2DConv2D3original_VGG19_B3_1/block2_conv1/Relu:activations:0>original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block2_conv2/Conv2D?
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block2_conv2_biasadd_readvariableop_resource6^original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block2_conv2/BiasAddBiasAdd0original_VGG19_B3_1/block2_conv2/Conv2D:output:0?original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2*
(original_VGG19_B3_1/block2_conv2/BiasAdd?
%original_VGG19_B3_1/block2_conv2/ReluRelu1original_VGG19_B3_1/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2'
%original_VGG19_B3_1/block2_conv2/Relu?
'original_VGG19_B3_1/block2_pool/MaxPoolMaxPool3original_VGG19_B3_1/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2)
'original_VGG19_B3_1/block2_pool/MaxPool?
.original_VGG19_B3_1/block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      20
.original_VGG19_B3_1/block3_conv1/dilation_rate?
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOpReadVariableOp=original_vgg19_b3_block3_conv1_conv2d_readvariableop_resource5^original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp*(
_output_shapes
:??*
dtype028
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp?
'original_VGG19_B3_1/block3_conv1/Conv2DConv2D0original_VGG19_B3_1/block2_pool/MaxPool:output:0>original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2)
'original_VGG19_B3_1/block3_conv1/Conv2D?
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>original_vgg19_b3_block3_conv1_biasadd_readvariableop_resource6^original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp*
_output_shapes	
:?*
dtype029
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp?
(original_VGG19_B3_1/block3_conv1/BiasAddBiasAdd0original_VGG19_B3_1/block3_conv1/Conv2D:output:0?original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2*
(original_VGG19_B3_1/block3_conv1/BiasAdd?
%original_VGG19_B3_1/block3_conv1/ReluRelu1original_VGG19_B3_1/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2'
%original_VGG19_B3_1/block3_conv1/Relu?
$mean_squared_error/SquaredDifferenceSquaredDifference3original_VGG19_B3_1/block3_conv1/Relu:activations:01original_VGG19_B3/block3_conv1/Relu:activations:0*
T0*0
_output_shapes
:?????????``?2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????``2
mean_squared_error/Mean?
'mean_squared_error/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'mean_squared_error/weighted_loss/Cast/x?
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 2W
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shape?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rank?
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:2V
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shape?
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :2U
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rank?
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 2e
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeShape mean_squared_error/Mean:output:0d^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape?
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const?
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
T0*+
_output_shapes
:?????????``2>
<mean_squared_error/weighted_loss/broadcast_weights/ones_like?
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*+
_output_shapes
:?????????``24
2mean_squared_error/weighted_loss/broadcast_weights?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*+
_output_shapes
:?????????``2&
$mean_squared_error/weighted_loss/Mul?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(mean_squared_error/weighted_loss/Const_1?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
mul/xn
mulMulmul/x:output:0*mean_squared_error/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul?
IdentityIdentityBlock_D1/BiasAdd:output:0 ^Block_D1/BiasAdd/ReadVariableOp^Block_D1/Conv2D/ReadVariableOp6^original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp6^original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp5^original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp6^original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp6^original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp5^original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp6^original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp5^original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp8^original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp7^original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp*^sequential/block10/BiasAdd/ReadVariableOp)^sequential/block10/Conv2D/ReadVariableOp*^sequential/block20/BiasAdd/ReadVariableOp)^sequential/block20/Conv2D/ReadVariableOp*^sequential/block21/BiasAdd/ReadVariableOp)^sequential/block21/Conv2D/ReadVariableOp*^sequential/block22/BiasAdd/ReadVariableOp)^sequential/block22/Conv2D/ReadVariableOp*^sequential/block30/BiasAdd/ReadVariableOp)^sequential/block30/Conv2D/ReadVariableOp3^sequential/conv2d_transpose/BiasAdd/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2B
Block_D1/BiasAdd/ReadVariableOpBlock_D1/BiasAdd/ReadVariableOp2@
Block_D1/Conv2D/ReadVariableOpBlock_D1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block1_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block1_conv1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp5original_VGG19_B3/block1_conv2/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp4original_VGG19_B3/block1_conv2/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block2_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block2_conv1/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp5original_VGG19_B3/block2_conv2/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp4original_VGG19_B3/block2_conv2/Conv2D/ReadVariableOp2n
5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp5original_VGG19_B3/block3_conv1/BiasAdd/ReadVariableOp2l
4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp4original_VGG19_B3/block3_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block1_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block1_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block1_conv2/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp6original_VGG19_B3_1/block1_conv2/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block2_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block2_conv1/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block2_conv2/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp6original_VGG19_B3_1/block2_conv2/Conv2D/ReadVariableOp2r
7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp7original_VGG19_B3_1/block3_conv1/BiasAdd/ReadVariableOp2p
6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp6original_VGG19_B3_1/block3_conv1/Conv2D/ReadVariableOp2V
)sequential/block10/BiasAdd/ReadVariableOp)sequential/block10/BiasAdd/ReadVariableOp2T
(sequential/block10/Conv2D/ReadVariableOp(sequential/block10/Conv2D/ReadVariableOp2V
)sequential/block20/BiasAdd/ReadVariableOp)sequential/block20/BiasAdd/ReadVariableOp2T
(sequential/block20/Conv2D/ReadVariableOp(sequential/block20/Conv2D/ReadVariableOp2V
)sequential/block21/BiasAdd/ReadVariableOp)sequential/block21/BiasAdd/ReadVariableOp2T
(sequential/block21/Conv2D/ReadVariableOp(sequential/block21/Conv2D/ReadVariableOp2V
)sequential/block22/BiasAdd/ReadVariableOp)sequential/block22/BiasAdd/ReadVariableOp2T
(sequential/block22/Conv2D/ReadVariableOp(sequential/block22/Conv2D/ReadVariableOp2V
)sequential/block30/BiasAdd/ReadVariableOp)sequential/block30/BiasAdd/ReadVariableOp2T
(sequential/block30/Conv2D/ReadVariableOp(sequential/block30/Conv2D/ReadVariableOp2h
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:, (
&
_user_specified_nameinput_tensor
?
?
+__inference_block3__net_layer_call_fn_44401
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3__net_layer_call_and_return_conditional_losses_438722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameinput_tensor
?
?
1__inference_original_VGG19_B3_layer_call_fn_43038
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
??
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44615

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
1__inference_original_VGG19_B3_layer_call_fn_44523

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_430612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
'__inference_block10_layer_call_fn_43263

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block10_layer_call_and_return_conditional_losses_432552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_block3__net_layer_call_fn_43901
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3__net_layer_call_and_return_conditional_losses_438722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
??
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44569

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_42877

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?'
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43061

inputs/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_428422&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_428632&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_428772
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_428962&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_429172&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_429312
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_429502&
$block3_conv1/StatefulPartitionedCall?
IdentityIdentity-block3_conv1/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_42842

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
΍
?

E__inference_sequential_layer_call_and_return_conditional_losses_44839

inputs*
&block30_conv2d_readvariableop_resource+
'block30_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource*
&block20_conv2d_readvariableop_resource+
'block20_biasadd_readvariableop_resource*
&block21_conv2d_readvariableop_resource+
'block21_biasadd_readvariableop_resource*
&block22_conv2d_readvariableop_resource+
'block22_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource*
&block10_conv2d_readvariableop_resource+
'block10_biasadd_readvariableop_resource
identity??block10/BiasAdd/ReadVariableOp?block10/Conv2D/ReadVariableOp?block20/BiasAdd/ReadVariableOp?block20/Conv2D/ReadVariableOp?block21/BiasAdd/ReadVariableOp?block21/Conv2D/ReadVariableOp?block22/BiasAdd/ReadVariableOp?block22/Conv2D/ReadVariableOp?block30/BiasAdd/ReadVariableOp?block30/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
block30/Conv2D/ReadVariableOpReadVariableOp&block30_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block30/Conv2D/ReadVariableOp?
block30/Conv2DConv2Dinputs%block30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block30/Conv2D?
block30/BiasAdd/ReadVariableOpReadVariableOp'block30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block30/BiasAdd/ReadVariableOp?
block30/BiasAddBiasAddblock30/Conv2D:output:0&block30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block30/BiasAddy
block30/ReluRelublock30/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block30/Reluz
conv2d_transpose/ShapeShapeblock30/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack?
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1?
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0block30/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose/BiasAdd?
block20/Conv2D/ReadVariableOpReadVariableOp&block20_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block20/Conv2D/ReadVariableOp?
block20/Conv2DConv2D!conv2d_transpose/BiasAdd:output:0%block20/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block20/Conv2D?
block20/BiasAdd/ReadVariableOpReadVariableOp'block20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block20/BiasAdd/ReadVariableOp?
block20/BiasAddBiasAddblock20/Conv2D:output:0&block20/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block20/BiasAdd{
block20/ReluRelublock20/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block20/Relu?
block21/Conv2D/ReadVariableOpReadVariableOp&block21_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block21/Conv2D/ReadVariableOp?
block21/Conv2DConv2Dblock20/Relu:activations:0%block21/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block21/Conv2D?
block21/BiasAdd/ReadVariableOpReadVariableOp'block21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block21/BiasAdd/ReadVariableOp?
block21/BiasAddBiasAddblock21/Conv2D:output:0&block21/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block21/BiasAdd{
block21/ReluRelublock21/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block21/Relu?
block22/Conv2D/ReadVariableOpReadVariableOp&block22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
block22/Conv2D/ReadVariableOp?
block22/Conv2DConv2Dblock21/Relu:activations:0%block22/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block22/Conv2D?
block22/BiasAdd/ReadVariableOpReadVariableOp'block22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
block22/BiasAdd/ReadVariableOp?
block22/BiasAddBiasAddblock22/Conv2D:output:0&block22/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block22/BiasAdd{
block22/ReluRelublock22/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block22/Relu~
conv2d_transpose_1/ShapeShapeblock22/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack?
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1?
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0block22/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_1/BiasAdd?
block10/Conv2D/ReadVariableOpReadVariableOp&block10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
block10/Conv2D/ReadVariableOp?
block10/Conv2DConv2D#conv2d_transpose_1/BiasAdd:output:0%block10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block10/Conv2D?
block10/BiasAdd/ReadVariableOpReadVariableOp'block10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
block10/BiasAdd/ReadVariableOp?
block10/BiasAddBiasAddblock10/Conv2D:output:0&block10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block10/BiasAddz
block10/ReluRelublock10/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block10/Relu?
IdentityIdentityblock10/Relu:activations:0^block10/BiasAdd/ReadVariableOp^block10/Conv2D/ReadVariableOp^block20/BiasAdd/ReadVariableOp^block20/Conv2D/ReadVariableOp^block21/BiasAdd/ReadVariableOp^block21/Conv2D/ReadVariableOp^block22/BiasAdd/ReadVariableOp^block22/Conv2D/ReadVariableOp^block30/BiasAdd/ReadVariableOp^block30/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2@
block10/BiasAdd/ReadVariableOpblock10/BiasAdd/ReadVariableOp2>
block10/Conv2D/ReadVariableOpblock10/Conv2D/ReadVariableOp2@
block20/BiasAdd/ReadVariableOpblock20/BiasAdd/ReadVariableOp2>
block20/Conv2D/ReadVariableOpblock20/Conv2D/ReadVariableOp2@
block21/BiasAdd/ReadVariableOpblock21/BiasAdd/ReadVariableOp2>
block21/Conv2D/ReadVariableOpblock21/Conv2D/ReadVariableOp2@
block22/BiasAdd/ReadVariableOpblock22/BiasAdd/ReadVariableOp2>
block22/Conv2D/ReadVariableOpblock22/Conv2D/ReadVariableOp2@
block30/BiasAdd/ReadVariableOpblock30/BiasAdd/ReadVariableOp2>
block30/Conv2D/ReadVariableOpblock30/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
1__inference_original_VGG19_B3_layer_call_fn_44630

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_434722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
,__inference_block1_conv2_layer_call_fn_42871

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_428632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_43359
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_433422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?+
?
E__inference_sequential_layer_call_and_return_conditional_losses_43314
input_1*
&block30_statefulpartitionedcall_args_1*
&block30_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_2*
&block20_statefulpartitionedcall_args_1*
&block20_statefulpartitionedcall_args_2*
&block21_statefulpartitionedcall_args_1*
&block21_statefulpartitionedcall_args_2*
&block22_statefulpartitionedcall_args_1*
&block22_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_2*
&block10_statefulpartitionedcall_args_1*
&block10_statefulpartitionedcall_args_2
identity??block10/StatefulPartitionedCall?block20/StatefulPartitionedCall?block21/StatefulPartitionedCall?block22/StatefulPartitionedCall?block30/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?
block30/StatefulPartitionedCallStatefulPartitionedCallinput_1&block30_statefulpartitionedcall_args_1&block30_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block30_layer_call_and_return_conditional_losses_430872!
block30/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(block30/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_431292*
(conv2d_transpose/StatefulPartitionedCall?
block20/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0&block20_statefulpartitionedcall_args_1&block20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block20_layer_call_and_return_conditional_losses_431502!
block20/StatefulPartitionedCall?
block21/StatefulPartitionedCallStatefulPartitionedCall(block20/StatefulPartitionedCall:output:0&block21_statefulpartitionedcall_args_1&block21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block21_layer_call_and_return_conditional_losses_431712!
block21/StatefulPartitionedCall?
block22/StatefulPartitionedCallStatefulPartitionedCall(block21/StatefulPartitionedCall:output:0&block22_statefulpartitionedcall_args_1&block22_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block22_layer_call_and_return_conditional_losses_431922!
block22/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(block22/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_432342,
*conv2d_transpose_1/StatefulPartitionedCall?
block10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0&block10_statefulpartitionedcall_args_1&block10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block10_layer_call_and_return_conditional_losses_432552!
block10/StatefulPartitionedCall?
IdentityIdentity(block10/StatefulPartitionedCall:output:0 ^block10/StatefulPartitionedCall ^block20/StatefulPartitionedCall ^block21/StatefulPartitionedCall ^block22/StatefulPartitionedCall ^block30/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2B
block10/StatefulPartitionedCallblock10/StatefulPartitionedCall2B
block20/StatefulPartitionedCallblock20/StatefulPartitionedCall2B
block21/StatefulPartitionedCallblock21/StatefulPartitionedCall2B
block22/StatefulPartitionedCallblock22/StatefulPartitionedCall2B
block30/StatefulPartitionedCallblock30/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
G
+__inference_block2_pool_layer_call_fn_42937

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_429312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
B__inference_block30_layer_call_and_return_conditional_losses_43087

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
+__inference_block3__net_layer_call_fn_43814
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block3__net_layer_call_and_return_conditional_losses_437852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
0__inference_conv2d_transpose_layer_call_fn_43137

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_431292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
B__inference_block22_layer_call_and_return_conditional_losses_43192

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_43234

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
:@?*
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
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?+
?
E__inference_sequential_layer_call_and_return_conditional_losses_43342

inputs*
&block30_statefulpartitionedcall_args_1*
&block30_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_2*
&block20_statefulpartitionedcall_args_1*
&block20_statefulpartitionedcall_args_2*
&block21_statefulpartitionedcall_args_1*
&block21_statefulpartitionedcall_args_2*
&block22_statefulpartitionedcall_args_1*
&block22_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_2*
&block10_statefulpartitionedcall_args_1*
&block10_statefulpartitionedcall_args_2
identity??block10/StatefulPartitionedCall?block20/StatefulPartitionedCall?block21/StatefulPartitionedCall?block22/StatefulPartitionedCall?block30/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?
block30/StatefulPartitionedCallStatefulPartitionedCallinputs&block30_statefulpartitionedcall_args_1&block30_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:?????????``?*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block30_layer_call_and_return_conditional_losses_430872!
block30/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(block30/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_431292*
(conv2d_transpose/StatefulPartitionedCall?
block20/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0&block20_statefulpartitionedcall_args_1&block20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block20_layer_call_and_return_conditional_losses_431502!
block20/StatefulPartitionedCall?
block21/StatefulPartitionedCallStatefulPartitionedCall(block20/StatefulPartitionedCall:output:0&block21_statefulpartitionedcall_args_1&block21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block21_layer_call_and_return_conditional_losses_431712!
block21/StatefulPartitionedCall?
block22/StatefulPartitionedCallStatefulPartitionedCall(block21/StatefulPartitionedCall:output:0&block22_statefulpartitionedcall_args_1&block22_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block22_layer_call_and_return_conditional_losses_431922!
block22/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(block22/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_432342,
*conv2d_transpose_1/StatefulPartitionedCall?
block10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0&block10_statefulpartitionedcall_args_1&block10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_block10_layer_call_and_return_conditional_losses_432552!
block10/StatefulPartitionedCall?
IdentityIdentity(block10/StatefulPartitionedCall:output:0 ^block10/StatefulPartitionedCall ^block20/StatefulPartitionedCall ^block21/StatefulPartitionedCall ^block22/StatefulPartitionedCall ^block30/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????``?::::::::::::::2B
block10/StatefulPartitionedCallblock10/StatefulPartitionedCall2B
block20/StatefulPartitionedCallblock20/StatefulPartitionedCall2B
block21/StatefulPartitionedCallblock21/StatefulPartitionedCall2B
block22/StatefulPartitionedCallblock22/StatefulPartitionedCall2B
block30/StatefulPartitionedCallblock30/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?'
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43025

inputs/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_428422&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_428632&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_428772
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_428962&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_429172&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_429312
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_429502&
$block3_conv1/StatefulPartitionedCall?
IdentityIdentity-block3_conv1/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
,__inference_block2_conv1_layer_call_fn_42904

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_428962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_42950

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
,__inference_block1_conv1_layer_call_fn_42850

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_428422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?'
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_42980
input_1/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_428422&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_428632&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_428772
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_428962&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_429172&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_429312
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_429502&
$block3_conv1/StatefulPartitionedCall?
IdentityIdentity-block3_conv1/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:+???????????????????????????::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
??
?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43518

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?
block1_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv1/dilation_rate?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
block1_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block1_conv2/dilation_rate?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
block2_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv1/dilation_rate?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
block2_conv2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block2_conv2/dilation_rate?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????``?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
block3_conv1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
block3_conv1/dilation_rate?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????``?2
block3_conv1/Relu?
IdentityIdentityblock3_conv1/Relu:activations:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????``?2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????F
output_1:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
encoder_Model
decoder_Model
conv_r1
	optimizer
_training_endpoints
regularization_losses
trainable_variables
	variables
		keras_api


signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Block3_Net", "name": "block3__net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Block3_Net"}, "training_config": {"loss": "specloss", "metrics": ["mse"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?C
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?@
_tf_keras_model?@{"class_name": "Model", "name": "original_VGG19_B3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "original_VGG19_B3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, null, null, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block3_conv1", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "original_VGG19_B3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, null, null, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block3_conv1", 0, 0]]}}}
?C
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?A
_tf_keras_sequential?A{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "block30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "block20", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "block21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "block22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "block10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "block30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "block20", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "block21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "block22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "block10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Block_D1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null, null, 64], "config": {"name": "Block_D1", "trainable": true, "batch_input_shape": [null, null, null, 64], "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?
(iter

)beta_1

*beta_2
	+decay
,learning_rate"m?#m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?"v?#v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
"14
#15"
trackable_list_wrapper
?
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
-10
.11
/12
013
114
215
316
417
518
619
720
821
922
:23
"24
#25"
trackable_list_wrapper
?
regularization_losses

Elayers
Flayer_regularization_losses
Gmetrics
trainable_variables
Hnon_trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, null, null, 3], "config": {"batch_input_shape": [null, null, null, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

;kernel
<bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
?

=kernel
>bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
@bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?

Akernel
Bbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

Ckernel
Dbias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9"
trackable_list_wrapper
?
regularization_losses

elayers
flayer_regularization_losses
gmetrics
trainable_variables
hnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

-kernel
.bias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
?

/kernel
0bias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
?

1kernel
2bias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block20", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?

3kernel
4bias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?

5kernel
6bias
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block22", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?

7kernel
8bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?

9kernel
:bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13"
trackable_list_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13"
trackable_list_wrapper
?
regularization_losses
?layers
 ?layer_regularization_losses
?metrics
trainable_variables
?non_trainable_variables
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3@2block3__net/Block_D1/kernel
':%2block3__net/Block_D1/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
$regularization_losses
?layers
 ?layer_regularization_losses
?metrics
%trainable_variables
?non_trainable_variables
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
A:???2%block3__net/sequential/block30/kernel
2:0?2#block3__net/sequential/block30/bias
J:H??2.block3__net/sequential/conv2d_transpose/kernel
;:9?2,block3__net/sequential/conv2d_transpose/bias
A:???2%block3__net/sequential/block20/kernel
2:0?2#block3__net/sequential/block20/bias
A:???2%block3__net/sequential/block21/kernel
2:0?2#block3__net/sequential/block21/bias
A:???2%block3__net/sequential/block22/kernel
2:0?2#block3__net/sequential/block22/bias
K:I@?20block3__net/sequential/conv2d_transpose_1/kernel
<::@2.block3__net/sequential/conv2d_transpose_1/bias
?:=@@2%block3__net/sequential/block10/kernel
1:/@2#block3__net/sequential/block10/bias
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
f
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
Iregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Jtrainable_variables
?non_trainable_variables
K	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
Mregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Ntrainable_variables
?non_trainable_variables
O	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Rtrainable_variables
?non_trainable_variables
S	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
Uregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Vtrainable_variables
?non_trainable_variables
W	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
Yregularization_losses
?layers
 ?layer_regularization_losses
?metrics
Ztrainable_variables
?non_trainable_variables
[	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]regularization_losses
?layers
 ?layer_regularization_losses
?metrics
^trainable_variables
?non_trainable_variables
_	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
aregularization_losses
?layers
 ?layer_regularization_losses
?metrics
btrainable_variables
?non_trainable_variables
c	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
iregularization_losses
?layers
 ?layer_regularization_losses
?metrics
jtrainable_variables
?non_trainable_variables
k	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
mregularization_losses
?layers
 ?layer_regularization_losses
?metrics
ntrainable_variables
?non_trainable_variables
o	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
rtrainable_variables
?non_trainable_variables
s	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
uregularization_losses
?layers
 ?layer_regularization_losses
?metrics
vtrainable_variables
?non_trainable_variables
w	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
yregularization_losses
?layers
 ?layer_regularization_losses
?metrics
ztrainable_variables
?non_trainable_variables
{	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
}regularization_losses
?layers
 ?layer_regularization_losses
?metrics
~trainable_variables
?non_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?trainable_variables
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mse", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mse", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?trainable_variables
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
::8@2"Adam/block3__net/Block_D1/kernel/m
,:*2 Adam/block3__net/Block_D1/bias/m
F:D??2,Adam/block3__net/sequential/block30/kernel/m
7:5?2*Adam/block3__net/sequential/block30/bias/m
O:M??25Adam/block3__net/sequential/conv2d_transpose/kernel/m
@:>?23Adam/block3__net/sequential/conv2d_transpose/bias/m
F:D??2,Adam/block3__net/sequential/block20/kernel/m
7:5?2*Adam/block3__net/sequential/block20/bias/m
F:D??2,Adam/block3__net/sequential/block21/kernel/m
7:5?2*Adam/block3__net/sequential/block21/bias/m
F:D??2,Adam/block3__net/sequential/block22/kernel/m
7:5?2*Adam/block3__net/sequential/block22/bias/m
P:N@?27Adam/block3__net/sequential/conv2d_transpose_1/kernel/m
A:?@25Adam/block3__net/sequential/conv2d_transpose_1/bias/m
D:B@@2,Adam/block3__net/sequential/block10/kernel/m
6:4@2*Adam/block3__net/sequential/block10/bias/m
::8@2"Adam/block3__net/Block_D1/kernel/v
,:*2 Adam/block3__net/Block_D1/bias/v
F:D??2,Adam/block3__net/sequential/block30/kernel/v
7:5?2*Adam/block3__net/sequential/block30/bias/v
O:M??25Adam/block3__net/sequential/conv2d_transpose/kernel/v
@:>?23Adam/block3__net/sequential/conv2d_transpose/bias/v
F:D??2,Adam/block3__net/sequential/block20/kernel/v
7:5?2*Adam/block3__net/sequential/block20/bias/v
F:D??2,Adam/block3__net/sequential/block21/kernel/v
7:5?2*Adam/block3__net/sequential/block21/bias/v
F:D??2,Adam/block3__net/sequential/block22/kernel/v
7:5?2*Adam/block3__net/sequential/block22/bias/v
P:N@?27Adam/block3__net/sequential/conv2d_transpose_1/kernel/v
A:?@25Adam/block3__net/sequential/conv2d_transpose_1/bias/v
D:B@@2,Adam/block3__net/sequential/block10/kernel/v
6:4@2*Adam/block3__net/sequential/block10/bias/v
?2?
+__inference_block3__net_layer_call_fn_44370
+__inference_block3__net_layer_call_fn_43814
+__inference_block3__net_layer_call_fn_44401
+__inference_block3__net_layer_call_fn_43901?
???
FullArgSpec/
args'?$
jself
jinput_tensor

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
 __inference__wrapped_model_42829?
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
annotations? *0?-
+?(
input_1???????????
?2?
F__inference_block3__net_layer_call_and_return_conditional_losses_43726
F__inference_block3__net_layer_call_and_return_conditional_losses_44339
F__inference_block3__net_layer_call_and_return_conditional_losses_44140
F__inference_block3__net_layer_call_and_return_conditional_losses_43670?
???
FullArgSpec/
args'?$
jself
jinput_tensor

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
1__inference_original_VGG19_B3_layer_call_fn_43038
1__inference_original_VGG19_B3_layer_call_fn_43074
1__inference_original_VGG19_B3_layer_call_fn_44508
1__inference_original_VGG19_B3_layer_call_fn_44523
1__inference_original_VGG19_B3_layer_call_fn_44630
1__inference_original_VGG19_B3_layer_call_fn_44645?
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
?2?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43001
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44569
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_42980
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44493
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44447
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44615?
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
?2?
*__inference_sequential_layer_call_fn_44877
*__inference_sequential_layer_call_fn_43403
*__inference_sequential_layer_call_fn_44858
*__inference_sequential_layer_call_fn_43359?
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
E__inference_sequential_layer_call_and_return_conditional_losses_43314
E__inference_sequential_layer_call_and_return_conditional_losses_43289
E__inference_sequential_layer_call_and_return_conditional_losses_44839
E__inference_sequential_layer_call_and_return_conditional_losses_44742?
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
?2?
(__inference_Block_D1_layer_call_fn_43423?
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
C__inference_Block_D1_layer_call_and_return_conditional_losses_43415?
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
2B0
#__inference_signature_wrapper_43941input_1
?2?
,__inference_block1_conv1_layer_call_fn_42850?
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
2?/+???????????????????????????
?2?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_42842?
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
2?/+???????????????????????????
?2?
,__inference_block1_conv2_layer_call_fn_42871?
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
G__inference_block1_conv2_layer_call_and_return_conditional_losses_42863?
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
+__inference_block1_pool_layer_call_fn_42883?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block1_pool_layer_call_and_return_conditional_losses_42877?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_block2_conv1_layer_call_fn_42904?
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
G__inference_block2_conv1_layer_call_and_return_conditional_losses_42896?
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
,__inference_block2_conv2_layer_call_fn_42925?
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
G__inference_block2_conv2_layer_call_and_return_conditional_losses_42917?
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
+__inference_block2_pool_layer_call_fn_42937?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block2_pool_layer_call_and_return_conditional_losses_42931?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_block3_conv1_layer_call_fn_42958?
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
G__inference_block3_conv1_layer_call_and_return_conditional_losses_42950?
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
'__inference_block30_layer_call_fn_43095?
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
B__inference_block30_layer_call_and_return_conditional_losses_43087?
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
0__inference_conv2d_transpose_layer_call_fn_43137?
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
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_43129?
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
'__inference_block20_layer_call_fn_43158?
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
B__inference_block20_layer_call_and_return_conditional_losses_43150?
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
'__inference_block21_layer_call_fn_43179?
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
B__inference_block21_layer_call_and_return_conditional_losses_43171?
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
'__inference_block22_layer_call_fn_43200?
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
B__inference_block22_layer_call_and_return_conditional_losses_43192?
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
2__inference_conv2d_transpose_1_layer_call_fn_43242?
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
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_43234?
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
'__inference_block10_layer_call_fn_43263?
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
B__inference_block10_layer_call_and_return_conditional_losses_43255?
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
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
C__inference_Block_D1_layer_call_and_return_conditional_losses_43415?"#I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
(__inference_Block_D1_layer_call_fn_43423?"#I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
 __inference__wrapped_model_42829?;<=>?@ABCD-./0123456789:"#:?7
0?-
+?(
input_1???????????
? "=?:
8
output_1,?)
output_1????????????
B__inference_block10_layer_call_and_return_conditional_losses_43255?9:I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
'__inference_block10_layer_call_fn_43263?9:I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_42842?;<I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_block1_conv1_layer_call_fn_42850?;<I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_42863?=>I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_block1_conv2_layer_call_fn_42871?=>I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_block1_pool_layer_call_and_return_conditional_losses_42877?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block1_pool_layer_call_fn_42883?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_block20_layer_call_and_return_conditional_losses_43150?12J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
'__inference_block20_layer_call_fn_43158?12J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
B__inference_block21_layer_call_and_return_conditional_losses_43171?34J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
'__inference_block21_layer_call_fn_43179?34J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
B__inference_block22_layer_call_and_return_conditional_losses_43192?56J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
'__inference_block22_layer_call_fn_43200?56J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
G__inference_block2_conv1_layer_call_and_return_conditional_losses_42896??@I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_block2_conv1_layer_call_fn_42904??@I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
G__inference_block2_conv2_layer_call_and_return_conditional_losses_42917?ABJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_block2_conv2_layer_call_fn_42925?ABJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
F__inference_block2_pool_layer_call_and_return_conditional_losses_42931?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block2_pool_layer_call_fn_42937?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_block30_layer_call_and_return_conditional_losses_43087?-.J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
'__inference_block30_layer_call_fn_43095?-.J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
F__inference_block3__net_layer_call_and_return_conditional_losses_43670?;<=>?@ABCD-./0123456789:"#>?;
4?1
+?(
input_1???????????
p
? "??<
5?2
0+???????????????????????????
? ?
F__inference_block3__net_layer_call_and_return_conditional_losses_43726?;<=>?@ABCD-./0123456789:"#>?;
4?1
+?(
input_1???????????
p 
? "??<
5?2
0+???????????????????????????
? ?
F__inference_block3__net_layer_call_and_return_conditional_losses_44140?;<=>?@ABCD-./0123456789:"#C?@
9?6
0?-
input_tensor???????????
p
? "/?,
%?"
0???????????
? ?
F__inference_block3__net_layer_call_and_return_conditional_losses_44339?;<=>?@ABCD-./0123456789:"#C?@
9?6
0?-
input_tensor???????????
p 
? "/?,
%?"
0???????????
? ?
+__inference_block3__net_layer_call_fn_43814?;<=>?@ABCD-./0123456789:"#>?;
4?1
+?(
input_1???????????
p
? "2?/+????????????????????????????
+__inference_block3__net_layer_call_fn_43901?;<=>?@ABCD-./0123456789:"#>?;
4?1
+?(
input_1???????????
p 
? "2?/+????????????????????????????
+__inference_block3__net_layer_call_fn_44370?;<=>?@ABCD-./0123456789:"#C?@
9?6
0?-
input_tensor???????????
p
? "2?/+????????????????????????????
+__inference_block3__net_layer_call_fn_44401?;<=>?@ABCD-./0123456789:"#C?@
9?6
0?-
input_tensor???????????
p 
? "2?/+????????????????????????????
G__inference_block3_conv1_layer_call_and_return_conditional_losses_42950?CDJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
,__inference_block3_conv1_layer_call_fn_42958?CDJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_43234?78J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_conv2d_transpose_1_layer_call_fn_43242?78J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_43129?/0J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
0__inference_conv2d_transpose_layer_call_fn_43137?/0J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_42980?
;<=>?@ABCDR?O
H?E
;?8
input_1+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_43001?
;<=>?@ABCDR?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44447?
;<=>?@ABCDQ?N
G?D
:?7
inputs+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44493?
;<=>?@ABCDQ?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44569
;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p

 
? ".?+
$?!
0?????????``?
? ?
L__inference_original_VGG19_B3_layer_call_and_return_conditional_losses_44615
;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p 

 
? ".?+
$?!
0?????????``?
? ?
1__inference_original_VGG19_B3_layer_call_fn_43038?
;<=>?@ABCDR?O
H?E
;?8
input_1+???????????????????????????
p

 
? "3?0,?????????????????????????????
1__inference_original_VGG19_B3_layer_call_fn_43074?
;<=>?@ABCDR?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "3?0,?????????????????????????????
1__inference_original_VGG19_B3_layer_call_fn_44508?
;<=>?@ABCDQ?N
G?D
:?7
inputs+???????????????????????????
p

 
? "3?0,?????????????????????????????
1__inference_original_VGG19_B3_layer_call_fn_44523?
;<=>?@ABCDQ?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "3?0,?????????????????????????????
1__inference_original_VGG19_B3_layer_call_fn_44630r
;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p

 
? "!??????????``??
1__inference_original_VGG19_B3_layer_call_fn_44645r
;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p 

 
? "!??????????``??
E__inference_sequential_layer_call_and_return_conditional_losses_43289?-./0123456789:A?>
7?4
*?'
input_1?????????``?
p

 
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_43314?-./0123456789:A?>
7?4
*?'
input_1?????????``?
p 

 
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_44742?-./0123456789:@?=
6?3
)?&
inputs?????????``?
p

 
? "/?,
%?"
0???????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_44839?-./0123456789:@?=
6?3
)?&
inputs?????????``?
p 

 
? "/?,
%?"
0???????????@
? ?
*__inference_sequential_layer_call_fn_43359?-./0123456789:A?>
7?4
*?'
input_1?????????``?
p

 
? "2?/+???????????????????????????@?
*__inference_sequential_layer_call_fn_43403?-./0123456789:A?>
7?4
*?'
input_1?????????``?
p 

 
? "2?/+???????????????????????????@?
*__inference_sequential_layer_call_fn_44858?-./0123456789:@?=
6?3
)?&
inputs?????????``?
p

 
? "2?/+???????????????????????????@?
*__inference_sequential_layer_call_fn_44877?-./0123456789:@?=
6?3
)?&
inputs?????????``?
p 

 
? "2?/+???????????????????????????@?
#__inference_signature_wrapper_43941?;<=>?@ABCD-./0123456789:"#E?B
? 
;?8
6
input_1+?(
input_1???????????"=?:
8
output_1,?)
output_1???????????