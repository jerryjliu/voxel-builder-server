var http = require('http');
var express = require('express');
var bodyParser = require('body-parser');
var ndarray = require('ndarray');
var fs = require('fs');
var cp = require('child_process');
var exec = require('child_process').exec;
var path = require('path');

const PORT = 8000;

// given ndarray trim zero padding to crop to boundaries of voxel grid
function trim_voxels(voxels) {
  var shape = voxels.shape;
  var mini, maxi, minj, maxj, mink, maxk;
  mini = Number.MAX_SAFE_INTEGER;
  maxi = 0;
  minj = Number.MAX_SAFE_INTEGER;
  maxj = 0;
  mink = Number.MAX_SAFE_INTEGER;
  maxk = 0;
  for(var i = 0; i < shape[0]; i++) {
    for(var j = 0; j < shape[1]; j++) {
      for(var k = 0; k < shape[2]; k++) {
        if (voxels.get(i,j,k) == 1) {
          if (i < mini) {
            mini = i;
          }
          if (i > maxi) {
            maxi = i;
          }
          if (j < minj) {
            minj = j;
          }
          if (j > maxj) {
            maxj = j;
          }
          if (k < mink) {
            mink = k;
          }
          if (k > maxk) {
            maxk = k;
          }
        }        
      }
    }
  }
  var reslen = (maxi - mini + 1) * (maxj - minj + 1) * (maxk - mink + 1);
  var results = ndarray(new Array(reslen), [maxi-mini+1,maxj-minj+1,maxk-mink+1]);
  for(var i = mini; i <= maxi; i++) {
    for(var j = minj; j <= maxj; j++) {
      for(var k = mink; k <= maxk; k++) {
        results.set(i-mini,j-minj,k-mink, voxels.get(i,j,k));
      }
    }
  }
  return results;
}

function swap_voxels_axis(voxels) {
  var shape = voxels.shape;
  var newShapeArr = new Array(shape.length);
  newShapeArr[0] = shape[0];
  newShapeArr[1] = shape[2];
  newShapeArr[2] = shape[1];
  var result = ndarray(new Array(shape[0] * shape[1] * shape[2]), newShapeArr);
  for(var i = 0; i < newShapeArr[0]; i++) {
    for(var j = 0; j < newShapeArr[1]; j++) {
      for(var k = 0; k < newShapeArr[2]; k++) {
        result.set(i,j,k,voxels.get(i,k,j));
      }
    }
  }
  return result;
}

// helper function - return index of largest element in array
function maxIndex(arr) {
  var max = Number.MIN_VALUE;
  var maxIndex = 0;
  for(var i = 0; i < arr.length; i++) {
   if (arr[i] > max) {
     max = arr[i]; maxIndex = i;
   }
  }
  return maxIndex;
}

// scale voxel grid until the maximum dimension hits the maxSize boundary
function scale_voxels(voxels, maxSize) {
  var shape = voxels.shape;
  var maxDim = maxIndex(shape);
  console.log('maxDim: ' + maxDim);
  var scaleRatio = maxSize / shape[maxDim];
  console.log('scaleRatio: ' + scaleRatio);
  var newShapeArr = new Array(shape.length);
  var resDataLength = 1;
  for(var i = 0; i < newShapeArr.length; i++) {
    newShapeArr[i] = Math.floor(scaleRatio * shape[i]);
    resDataLength *= newShapeArr[i];
  }
  console.log('new shape arr');
  console.log(newShapeArr);
  console.log('resDataLength');
  console.log(resDataLength);
  var result = ndarray(new Array(resDataLength), newShapeArr);
  for(var i = 0; i < newShapeArr[0]; i++) {
    for(var j = 0; j < newShapeArr[1]; j++) {
      for(var k = 0; k < newShapeArr[2]; k++) {
        var x,y,z; 
        x = Math.floor(i / scaleRatio);
        y = Math.floor(j / scaleRatio);
        z = Math.floor(k / scaleRatio);
        //console.log(x + " " + y + " " + z + " " + voxels.get(x,y,z));
        result.set(i,j,k,voxels.get(x,y,z));
      }
    }
  }
  return result;
}

// pad voxel grid to a cube, centering the voxels
function pad_voxels(voxels) {
  var shape = voxels.shape;
  var maxDim = maxIndex(shape);
  var newShapeArr = new Array(shape.length);
  var resDataLength = 1;
  for(var i = 0; i < newShapeArr.length; i++) {
    newShapeArr[i] = shape[maxDim];
    resDataLength *= newShapeArr[i];
  }
  var result = ndarray(new Array(resDataLength), newShapeArr);
  for(var i = 0; i < newShapeArr[0]; i++) {
    var disti = i - Math.floor(newShapeArr[0]/2);
    var oldi = Math.floor(shape[0]/2) + disti;
    for(var j = 0; j < newShapeArr[1]; j++) {
      var distj = j - Math.floor(newShapeArr[1]/2);
      var oldj = Math.floor(shape[1]/2) + distj;
      for(var k = 0; k < newShapeArr[2]; k++) {
        var distk = k - Math.floor(newShapeArr[2]/2);
        var oldk = Math.floor(shape[2]/2) + distk;
        var newval;
        if (oldi < 0 || oldi >= shape[0] || oldj < 0 || oldj >= shape[1] || oldk < 0 || oldk >= shape[2]) {
          newval = 0;
        } else {
          newval = voxels.get(oldi, oldj, oldk);
        }
        result.set(i,j,k,newval);
      }
    }
  }
  return result;
}

// convert 3D ndarray into ascii Torch .t7 file for output
function ndarray_to_t7(voxels) {
  var invoxels_content = ""; // make sure obeys torch format
  invoxels_content += "4\n1\n3\nV 1\n18\ntorch.DoubleTensor\n";
  invoxels_content += "4\n";  // dimension of input tensor
  invoxels_content += "1 64 64 64\n";
  invoxels_content += "262144 4096 64 1\n";
  invoxels_content += "1\n";
  invoxels_content += "4\n";
  invoxels_content += "2\n";
  invoxels_content += "3\n";
  invoxels_content += "V 1\n";
  invoxels_content += "19\n";
  invoxels_content += "torch.DoubleStorage\n";
  invoxels_content += "262144\n";
   //print contents of padded voxel cube to file
  for(var i = 0; i < voxels.data.length; i++) {
    invoxels_content += voxels.data[i] + " ";
  }
  return invoxels_content;
}

// convert ascii Torch .t7 file to 3D ndarray
function t7_to_ndarray(tfile) {
  var contents = fs.readFileSync(tfile).toString();
  var lines = contents.split(/\r?\n/);
  var ndim = 3;
  var shapeArr = new Array(ndim);
  var strideArr = new Array(ndim);
  var numel;
  var dataArr;
  for(var i = 0; i < lines.length; i++) {
   if (i <= 6) continue; 
   else if (i == 7) {
     var tokens = lines[i].split(/[ ,]+/);
     var start = 1;
     if (tokens.length == 5) start = 2; 
     for(var j = start; j < tokens.length; j++) {
       shapeArr[j - start] = parseInt(tokens[j]);
     }
   } else if (i == 8) {
     var tokens = lines[i].split(/[ ,]+/);
     var start = 1;
     if (tokens.length == 5) start = 2; 
     for(var j = start; j < tokens.length; j++) {
       strideArr[j - start] = parseInt(tokens[j]);
     }
   } else if (i >= 9 && i <= 15) {
     continue;
   } else if (i == 16) {
     numel = parseInt(lines[i]); 
     dataArr = new Array(numel);
   } else if (i >= 17) {
     var tokens = lines[i].split(/[ ,]+/);
     for(var j = 0; j < tokens.length; j++) {
       dataArr[j] = parseInt(tokens[j]);
     }
   }
  }
  console.log(shapeArr);
  console.log(strideArr);
  var result = ndarray(dataArr, shapeArr, strideArr);
  return result;
}

var app = express();
app.use(bodyParser.json());
app.get('/', function(req, res) {
  //var invoxels_file = "tmp/testvoxels.t7";
  //var invoxels_content = "sdasdfasdfadf"; // make sure obeys torch format
  //console.log(invoxels_content);
  //console.log(__dirname);
  //fs.writeFileSync(invoxels_file, invoxels_content, "utf8");
  //var tmptest = fs.readFileSync(invoxels_file, "utf8");
  //console.log(tmptest);
  //exec('th -h', function(error, stdout, stderr) {
    //console.log(stdout);
    //console.log("errors: " + stderr);
  //});

  res.send('Hello World!\n');
});
app.post('/process', function(request, response) {
  var invoxels = request.body;
  invoxels = ndarray(invoxels.data, invoxels.shape, invoxels.stride, invoxels.offset);
  console.log(invoxels);

  // process voxels
  var invoxels2 = trim_voxels(invoxels);
  console.log('TRIMMED VOXELS:');
  console.log(invoxels2);
  // swap y-z axis
  var invoxels2_flip = swap_voxels_axis(invoxels2);
  // scale voxels until one end hits 64
  var invoxels3 = scale_voxels(invoxels2_flip, 64);
  console.log('SCALED VOXELS:');
  console.log(invoxels3);
  // pad voxel grid to create a cube with length of all sides equal to length of maximum dimension
  var invoxels4 = pad_voxels(invoxels3);

  // export voxels to a tensor file
  var invoxels_file = "testvoxels"; var invoxels_ext = ".t7";
  var data_dir = '/data/jjliu/models';
  var proj_inputs_dir = 'proj_inputs_voxel';
  var full_invoxels_file = path.join(data_dir, proj_inputs_dir, invoxels_file + invoxels_ext);
  //var invoxels_content = "sdasdfasdfadf"; // make sure obeys torch format
  var invoxels_content = ndarray_to_t7(invoxels4);

  fs.writeFileSync(full_invoxels_file, invoxels_content);

  // run torch script on file
  var outvoxels_file = invoxels_file + "_out";
  var outvoxels_ext = ".t7";
  var proj_outputs_dir = 'proj_outputs_voxel';
  var full_outvoxels_file = path.join(data_dir, proj_outputs_dir, outvoxels_file + outvoxels_ext);
  var cmd = "th ";
  var proj_file = "/home/jjliu/Documents/3dexps/proj_generate.lua";
  cmd += proj_file + " ";
  var gpu_opt = "-gpu 2";
  cmd += gpu_opt + " ";
  var voxel_opt = "-input " + invoxels_file;
  cmd += voxel_opt + " ";
  var informat_opt = "-informat t7";
  cmd += informat_opt + " ";
  var ckp_opt = "-ckp checkpoints_64chair100o_vaen2";
  cmd += ckp_opt + " ";
  var ckgen_opt = "-ckgen 1450";
  cmd += ckgen_opt + " ";
  var ckproj_opt = "-ckproj 180";
  cmd += ckproj_opt + " ";
  var ckext_opt = "-ckext feat4";
  cmd += ckext_opt + " ";
  //TODO: implement -out in Torch
  var outf_opt = "-out " + path.join(data_dir, proj_outputs_dir, outvoxels_file);
  cmd += outf_opt + " ";
  var outformat_opt = "-outformat t7";
  cmd += outformat_opt + " ";

  console.log(cmd);
  //cp.execSync(cmd);     

  // read from outvoxels 
  console.log(full_outvoxels_file);
  var outvoxels = t7_to_ndarray(full_outvoxels_file);
  //console.log(outvoxels);
  console.log(outvoxels.shape);
  // flip y,z axes of outvoxels
  var outvoxels_flip = swap_voxels_axis(outvoxels);
  // return this file / figure out how to display it on the client
  //console.log(outvoxels_flip);
  //console.log(outvoxels_flip.shape);
  var outvoxels2 = trim_voxels(outvoxels_flip);
  //console.log(outvoxels2);
  console.log(outvoxels2.shape);
  response.json(outvoxels2);
});
app.listen(PORT);
