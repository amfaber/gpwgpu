struct PushConstants{
    n_elements: u32,
	#EXTRAPUSHCONSTANTS
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: f32;

#EXTRABUFFERS

var<push_constant> pc: PushConstants;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	// Not checking if we are outside the alloted amount of work, as only 1 invocation of this function should ever be started.
	// i.e. workgroup_size(1, 1, 1), dispatcher(1, 1, 1)
	var datum = input[0];
    var acc = datum;
	
	for (var i = 1u; i < pc.n_elements; i++){
		datum = input[i];
		#OPERATION
	}
	#EXTRALAST
    output = acc;
}


