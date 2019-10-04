import Vue from 'vue'
import App from './App.vue'
import locale from 'element-ui/lib/locale/lang/en'
// import VuejsClipper from 'vuejs-clipper'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';


// Vue.use(VuejsClipper)
Vue.use(ElementUI, { locale });
Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app')
