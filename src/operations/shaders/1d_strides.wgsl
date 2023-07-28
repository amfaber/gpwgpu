struct PushConstants{
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

	var coords: array<i32, #N>;

	var strides: array<i32, #N>;

	var next_stride = 1;

	
	#for I in 0..N{
		strides[#N - 1 - #I] = next_stride;
		next_stride *= pc.bounds[#N - 1 - #I];
	
	}
	
	if global_id.x >= u32(next_stride){
		return;
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
	
	let rint = #RINT_EXPR;
	
	#INIT
	
	for (var i = -rint; i <= rint; i += 1){
		if (coords[#N - 1] + i >= 0) && (coords[#N - 1] + i < pc.bounds[#N - 1]){
			#KERNEL_FUNC
			#if NORMALIZE{
				kernel_acc += kernel_eval;
			}
		} else {
			#BOUNDARY
		}
		
	}

	#POST

	#if NORMALIZE{
		acc /= kernel_acc;
	}
	
	next_stride = 1;
	idx = 0;
	#for I in 0..N-1{
		idx += next_stride * coords[#N - 2 - #I];
		next_stride *= pc.bounds[#N - 2 - #I];
	}
	idx += coords[#N - 1] * next_stride;
	
	#OUTPUT // Usually something like output[idx] = acc;
}

