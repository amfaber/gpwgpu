#export get_flat_idx{
	fn get_flat_idx(wg_id: vec3<u32>, wg_num: vec3<u32>, local_index: u32) -> u32{
		let wg_grid_flat = (wg_id.x + wg_id.y * wg_num.x + wg_id.z * wg_num.x * wg_num.y);
		let wg_total = #WG_X * #WG_Y * #WG_Z;
		return  wg_grid_flat * wg_total + local_index;
	}
}

