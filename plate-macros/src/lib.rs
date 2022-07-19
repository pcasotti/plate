use quote::quote;

#[proc_macro_derive(Vertex, attributes(vertex))]
pub fn vert(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let ident = &ast.ident;

    let fields = if let syn::Data::Struct(syn::DataStruct { fields, .. }) = &ast.data {
        if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = fields {
            named
        } else { unimplemented!("unnamed field") }
    } else { unimplemented!("not struct") };

    let attributed = fields.iter()
        .filter_map(|f| {
            let attr = f.attrs.iter()
                .find(|a| {
                    a.path.segments.iter()
                        .find(|s| s.ident == "vertex")
                        .is_some()
                });

            match attr {
                Some(a) => Some((f, a)),
                None => None,
            }
        });

    let attribute_descriptions = attributed
        .map(|(f, attr)| {
            let ident = f.ident.as_ref().unwrap();

            let meta = attr.parse_meta().unwrap();
            let location_nv = if let syn::Meta::List(p) = &meta {
                p.nested.iter()
                    .find_map(|nm| {
                        if let syn::NestedMeta::Meta(m) = nm {
                            if let syn::Meta::NameValue(nv) = m {
                                match nv.path.segments.iter().any(|s| s.ident == "loc") {
                                    true => Some(nv),
                                    false => None,
                                }
                            } else { unimplemented!("not namevalue") }
                        } else { unimplemented!("not nested meta") }
                    })
                    .expect("no attr?")
            } else { unreachable!() };
            let location = if let syn::Lit::Int(i) = &location_nv.lit {
                i
            } else { unimplemented!("not int") };

            let format_nv = if let syn::Meta::List(p) = &meta {
                p.nested.iter()
                    .find_map(|nm| {
                        if let syn::NestedMeta::Meta(m) = nm {
                            if let syn::Meta::NameValue(nv) = m {
                                match nv.path.segments.iter().any(|s| s.ident == "format") {
                                    true => Some(nv),
                                    false => None,
                                }
                            } else { unimplemented!("not namevalue") }
                        } else { unimplemented!("not nested meta") }
                    })
                    .expect("no attr?")
            } else { unreachable!() };
            let format = if let syn::Lit::Str(s) = &format_nv.lit {
                //syn::Ident::new(&s.value(), s.span())
                quote::format_ident!("{}", s.value())
            } else { unimplemented!("not str") };

            quote! {
                plate::VertexAttributeDescription::new(0, #location, memoffset::offset_of!(Self, #ident) as u32, plate::Format::#format)
            }
        });

    quote! {
        impl plate::VertexDescription for #ident {
            fn binding_descriptions() -> Vec<plate::VertexBindingDescription> {
                vec![
                    plate::VertexBindingDescription::new(0, std::mem::size_of::<Self>() as u32, plate::InputRate::VERTEX)
                ]
            }

            fn attribute_descriptions() -> Vec<plate::VertexAttributeDescription> {
                vec![
                    #(#attribute_descriptions),*
                ]
            }
        }
    }.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    //test not struct
    //unnamed field
    //non attributed field
    //wrong attribute field
    //other macros attributes
}
