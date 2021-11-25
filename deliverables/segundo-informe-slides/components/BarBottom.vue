<script>
export default {
  props: {
    title: {
      type: String,
      default: "Marco Antonio Vela Rodriguez",
    },
  }
}
</script>

<template>
  <div v-if="title || social" class="bg-barBottom flex absolute w-full bottom-0 left-0 py-0.6 px-4 text-peq">
    <div class="w-1/2 text-left">
      {{ title }}
    </div>
    <div class="w-1/2 flex justify-end">
      <SlideCurrentNo />  /  <SlidesTotal />
    </div>
  </div>
</template>

<style scoped>
.bg-barBottom {
    background-color: rgb(255, 191, 0);
}
.text-peq {
    font-size: 0.75rem;
}

</style>