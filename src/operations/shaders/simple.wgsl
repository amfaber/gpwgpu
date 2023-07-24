@group(0) @binding(0)
var<storage, read#if INPLACE{_write}> input: array<f32>;

#if BINARY{
	@group(0) @binding(1)
	var<storage, read> input2: array<f32>;
}

#if !INPLACE{
	@group(0) @binding(2)
	var<storage, read_write> output: array<f32>;
}

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	let idx = global_id.x;
	if (idx >= #LENGTH) {
		return;
	}
	let data = #OPERATION;
	#if INPLACE{
		input[idx] = data;
	} #else {
		output[idx] = data;
	}
}
