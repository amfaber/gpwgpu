// IDEAS
// Pass the current strides through push constants
// Dispatch completely flatly no matter what (n, 1, 1) and figure out accesses manually in the shader
// This allows a truely n-D kernel, which is specialized perfectly to the dimension at compile time
// As before, transpose the image on writing, allowing the first dim to always access with stride 1



struct PushConstants{
    curdim: u32,
	maxdim: u32,
	
	kernel_dims: i32,

	sigma: f32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<push_constant> pc: PushConstants;

var<workgroup> local_storage: array<f32, #LOCALSIZE>;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_index) local_index: u32,
) {
	let x = i32(global_id.x);
	let y = i32(global_id.y);
	let z = i32(global_id.z);

	var stride_y: i32;
	var stride_z: i32;

	#if NDIM == 2
	if pc.curdim == 0u{
		stride_y = #image_dims_j;
		// global_id.z will always be 0 in 2D, so this doesn't matter for access, but
		stride_z = 0;
	} else if pc.curdim == 1u{
		stride_y = #image_dims_i;
		stride_z = 0;
	}
	#endif
	
	#if NDIM == 3
	if pc.curdim == 0u{
		stride_y = #image_dims_k;
		stride_z = #image_dims_j * #image_dims_k;
	} else if pc.curdim == 1u{
		stride_y = #image_dims_i;
		stride_z = #image_dims_k * #image_dims_i;
	} else if pc.curdim == 2u{
		stride_y = #image_dims_j;
		stride_z = #image_dims_i * #image_dims_j;
	}
	#endif
	
	
	let offset = y * stride_y + z * stride_z;


	let rint = i32(pc.kernel_dims / 2);

	let local_idx = i32(local_index);
	var global_access = local_idx - rint + x + offset;
	var local_access = local_idx;
	var load_x = x - rint;

	while global_access < offset + #LOCALSIZE + rint{
		// stride_y will always be the bound of the axis we are accessing along
		if load_x < 0 || load_x >= stride_y {
			local_storage[local_access] = 0.0;
		} else {
			local_storage[local_access] = input[global_access];
		}
		local_access += #LOCALSIZE;
		global_access += #LOCALSIZE;
		load_x += #LOCALSIZE;
	}
	
	
	workgroupBarrier();
	
	if (x >= stride_y) {
		return;
	}

	let sigma2 = pc.sigma * pc.sigma;
	
	var acc = 0.;
	var gauss_sum = 0.;
	for (var i: i32 = -rint; i <= rint; i = i + 1){
		let i_float = f32(i);
		let gauss_eval = exp(-i_float*i_float / (2.0 * sigma2));
		acc += local_storage[local_idx + i] * gauss_eval;
		gauss_sum += gauss_eval;
	}

	acc /= gauss_sum;


	#if NDIM == 2
	let output_idx = x * stride_y + y;
	#endif
	
	#if NDIM == 3
	let output_idx = x * stride_z + y + z * stride_y;
	#endif
	
	output[output_idx] = acc;
	// output[x + offset] = acc;
}

