<template>
  <el-row :gutter="5">
    <el-col :span="6">
       <el-card class="box-card" :shadow='"never"' :body-style='body_style'>
        <div slot="header" class="clearfix">
          <span>Filter</span>
        </div>
        <div style="padding: 0px">
          <p>detail for search filter</p>
        </div>
      </el-card>
    </el-col>
    <el-col :span="10">
      <image-crop @listImageObject="getListImage"/>
    </el-col>
    <el-col :span="8" style="overflow-y:auto; height:97vh;">
      <el-card class="box-card" :shadow='"never"' :body-style='body_style' v-if="listImage != null">
        <div style="padding: 0px">
          <el-row>
            <el-col :span="12">
              <el-card :body-style="{ padding: '0px' }" v-for="(o, index) in listImage.filter((_,idx) => idx % 2 ==0 )" :key="index">
                <img :src="'http://127.0.0.1:5000/api/v1'+ o.ref" class="image">
                <div class="card-detail">
                  <strong style="font-size:14px">{{o.product}}</strong><br>
                  <span>{{o.brach}}</span>
                  <div class="bottom clearfix">
                    <el-button type="text" :size="'small'" >chi tiết</el-button>
                  </div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card :body-style="{ padding: '0px' }" v-for="(o, index) in listImage.filter((_,idx) => idx % 2 !=0 )" :key="index">
                <img :src="'http://127.0.0.1:5000/api/v1'+ o.ref" class="image">
                <div class="card-detail">
                  <strong style="font-size:14px">{{o.product}}</strong><br>
                  <span>{{o.brach}}</span>
                  <div class="bottom clearfix">
                    <el-button type="text" :size="'small'" >chi tiết</el-button>
                  </div>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>
      </el-card>
    </el-col>
</el-row>
</template>

<script>
import ImageCrop from './components/ImageCrop.vue';
export default {
  name: 'app',
  components: {
    ImageCrop,
  },
  data() {
    return {
      body_style:{ padding: '5px'},
      currentDate: new Date().getDate(),
      listImage: [
         {"ref": "/image/0_0_083.png", "brach": "Christian Louboutin", "product": "shoes"}
      ]
    }
  },
  methods: {
    /* eslint-disable */ 
    getListImage: function(value){
      this.listImage=[...JSON.parse(value)]
    }
  },
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  margin: 0px;
  padding: 0px;
}
.card-detail{
  padding: 10px 4px;
  font-size: 12px;
}
.el-card__header{
  padding:10px 8px !important; 
}
.text {
    font-size: 14px;
  }

  .item {
    margin-bottom: 0px;
  }

  .clearfix:before,
  .clearfix:after {
    display: table;
    content: "";
  }
  .clearfix:after {
    clear: both
  }

  .box-card {
    width: 100%;
  }
.image {
    width: 100%;
    display: block;
  }
</style>
