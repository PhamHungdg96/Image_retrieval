<template>
    <div>
        <clipper-upload v-model="src[1]">upload image</clipper-upload>
        <div class="flex">
            <clipper-basic class="basic lg" ref="clipper"
            :src="src[1]"
            :grid='false'
            :shadow='"rgba(255, 255, 255, 0.4)"'
            :touch-create='false'
            preview="preview"
            ></clipper-basic>
            <clipper-preview class="basic md" name="preview"></clipper-preview>
        </div>
    </div>
</template>
<script>
import { clipperBasic, clipperUpload,clipperPreview } from 'vuejs-clipper'
export default {
    name:'ImageClip',
    components:{clipperBasic, clipperUpload,clipperPreview},
    data(){
        return{
            src: [
            "https://timtnleeproject.github.io/vuejs-clipper/ex2.jpg",
            "https://timtnleeproject.github.io/vuejs-clipper/dawn.jpg"
            ],
        }
    },
    methods: {
    },
    created() {
        console.log()
        // this.$refs.clipper.onChange$.subscribe(() => {
        //     console.log('hung')
        // })
    },
}
</script>