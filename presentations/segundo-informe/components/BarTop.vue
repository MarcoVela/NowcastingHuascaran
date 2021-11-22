<script>
export default {
  props: {
    title: {
      type: String,
      required: true,
    },
  }
}
</script>

<template>
  <div v-if="title || social" class="bg-barTop flex absolute w-full top-0 left-0 py-3.5 px-4 text-big">
    <div class="text-left">
      {{ title }}
    </div>
  </div>
</template>

<style scoped>
.bg-barTop {
    background-color: rgb(255, 196, 0);
}
.text-big {
    font-size: 1.4rem;
    font-weight: bolder;
}
</style>