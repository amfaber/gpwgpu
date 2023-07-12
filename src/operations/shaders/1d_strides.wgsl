// IDEAS
// Pass the current strides through push constants
// Dispatch completely flatly no matter what (n, 1, 1) and figure out accesses manually in the shader
// This allows a truely n-D kernel, which is specialized perfectly to the dimension at compile time
// As before, transpose the image on writing, allowing the first dim to always access with stride 1



struct PushConstants{
	// strides: array<u32, #N>,
	bounds: array<i32, #N>,
	#EXTRA_PUSHCONSTANTS
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

#EXTRA_BUFFERS

var<push_constant> pc: PushConstants;

// var<workgroup> local_storage: array<f32, #LOCALSIZE>;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_index) local_index: u32,
) {
	var size = 1;
	#for I in 0..N{
		size *= pc.bounds[#I];
	}
	if global_id.x >= u32(size){
		return;
	}
	
	var coords: array<i32, #N>;

	var strides: array<i32, #N>;

	var next_stride = 1;

	#for I in 0..N{
		strides[#N - 1 - #I] = next_stride;
		next_stride *= pc.bounds[#I];
	
	}

	var idx = i32(global_id.x);

	#for I in 0..N{
		coords[#I] = idx / strides[#I];
		idx = idx % strides[#I];

	}

	idx = i32(global_id.x);
	
	var acc = 0.0;
	var kernel_eval = 0.0;
	#if NORMALIZE{
		var kernel_acc = 0.0;
	}
	
	#for I in -RINT..=RINT{
		if (coords[#N - 2] + #I >= 0) && (coords[#N - 2] + #I < pc.bounds[#N - 2]){
			kernel_eval = #KERNEL_FUNC;
			acc += input[idx + #I] * kernel_eval;
			#if NORMALIZE{
				kernel_acc += kernel_eval;
			}
		} else {
			#BOUNDARY
		}
		
	}

	#if NORMALIZE{
		acc /= kernel_acc;
	}
	
	// output[0] = f32(pc.bounds[0]);
	// output[1] = f32(pc.bounds[1]);
	
	idx = 0;
	#for I in 0..N - 1{
		
		idx += coords[#I] * strides[#I + 1];
	}
	idx += coords[#N - 1] * strides[0];
	
	output[idx] = acc;
	// output[0] = coords[0];
}

