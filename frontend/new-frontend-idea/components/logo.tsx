import Image from "next/image"

export function Logo() {
  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary">
      <Image src="/assets/logo.svg" alt="Logo" width={24} height={24} />
    </div>
  )
}

