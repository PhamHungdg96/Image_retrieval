<template>
  <div>
    <cropper
    classname="cropper"
    :src="img"
    :stencilProps="{
      linesClassnames:{
        default: 'line',
        
      },
      handlers:{
        eastNorth: true,
        north: false,
        westNorth: true,
        west: false,
        westSouth: true,
        south: false,
        eastSouth: true,
        east: false
      },
      handlersClassnames:{
        eastNorth: 'handler--east-north',
        westNorth: 'handler--west-north',
        westSouth: 'handler--west-south',
        eastSouth: 'handler--east-south',
      }
    }"
    @change="change"
  ></cropper>
    <div class="button-wrapper">
			<span class="button" @click="$refs.file.click()">
				<input type="file" ref="file" @change="uploadImage($event)" accept="image/*">
				Upload image
			</span>
		</div>
  </div>
  
</template>

<script>
/* eslint-disable */
import {Cropper} from 'vue-advanced-cropper'
import {Button} from 'element-ui'
import axios from 'axios'
export default {
  name: 'ImageCrop',
  props: {
    listImage:'sfsd'
  },
  components:{Cropper,Button},
  data() {
      return {img: null}
  },
  methods: {
    uploadImage(event) {
			var input = event.target;
			if (input.files && input.files[0]) {
					var reader = new FileReader();
					reader.onload = (e) => {
							this.img = e.target.result;
					}
					reader.readAsDataURL(input.files[0]);
			}
    },
    emit_val(value){
      this.$emit('listImageObject',value)
    },
    change({coordinates, canvas}) {
      if (this.img !=null){
        let data
        axios.post('http://127.0.0.1:5000/api/v1/image/',{
          "dataImage": canvas.toDataURL()
        }).then((response)=>{
          // console.log(response.data)
          this.emit_val(response.data)
        }).catch((e)=>{
          
          console.error(e)
          // console.log(coordinates, canvas.toDataURL())
        })
      }
      
    }
  },
}
</script>

<style scoped>
.cropper {
  height: 500px;
  /* border: 2px solid #38d890; */
  background-color: #555
}
.button-wrapper {
	display: flex;
	justify-content: center;
	margin-top: 17px;
}

.button {
	color: white;
	font-size: 14px;
	padding: 10px 20px;
	background: #3fb37f;
  cursor: pointer;
  border-radius: 5px;
	transition: background 0.5s;
}

.button:hover {
	background: #38d890;
}

.button input {
	display: none;
}

</style>
